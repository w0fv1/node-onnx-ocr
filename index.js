import fs from "fs";
import path from "path";
import sharp from "sharp";
import * as ort from "onnxruntime-node";
import os from "os";

/**
 * OCR 配置对象
 */
const DEFAULT_CONFIG = {
    // =========================================================================
    // 1. 模型与文件路径设置
    // =========================================================================
    detModel: "./model/det.mobile.onnx",
    recModel: "./model/rec.mobile.nhwc_uint8.onnx",
    dictPath: "./model/dict.txt",
    debugDir: "./debug",

    // =========================================================================
    // 2. 运行时与硬件设置
    // =========================================================================
    saveDebugImages: false,
    executionProviders: ["cpu"],
    detThreads: Math.max(1, Math.ceil(os.cpus().length / 2)),
    recThreads: Math.max(1, Math.ceil(os.cpus().length / 4)),
    batchSize: 16,

    // =========================================================================
    // 3. 图像预处理参数
    // =========================================================================
    physicalWidthMM: null,
    targetPPI: 156,
    detLimitSide: 960,

    // =========================================================================
    // 4. 文本检测 (Detection) 参数
    // =========================================================================
    detDbThresh: 0.3,
    unclipRatio: 1.2,
    boxPadding: 2,

    // 过滤与清洗
    filterStrength: "low", // 'none' | 'low' | 'medium' | 'high'
    removeDenseClusters: true,
    overlapRatio: 0.1,
    maxClusterSize: 3,

    // =========================================================================
    // 5. 文本识别 (Recognition) 参数
    // =========================================================================
    recImgH: 48,
    textMinLength: 2, // 识别结果短于此值将被标记为 invalid

    // 文本后处理
    removeEmoji: true,
    removeChars: [], // 例如 ['©', '®']

    // 排序与排版
    sortMode: "column", // 'raw' | 'top-down' | 'column'

    smartSpacing: true,
    baseBlankThresh: 3,
    shortBlockThresh: 5,
    shortBlockAspectRatio: 4.0,
    trashSparseThresh: 2,
};

/**
 * @typedef {Object} Box
 * @property {number} x
 * @property {number} y
 * @property {number} w
 * @property {number} h
 */

/**
 * @typedef {Object} OcrBlock
 * @property {Box} box - 坐标信息
 * @property {string} status - 'pending' | 'valid' | 'rejected'
 * @property {string} [rejectReason] - 被拒绝的原因 (如: 'shape_filter', 'empty_text', 'cluster')
 * @property {string} rawText - 模型识别出的原始文本
 * @property {string} text - 清洗后的最终文本
 * @property {number} score - (预留) 置信度
 * @property {number} sortIndex - 最终排序后的索引
 */

class Ocr {
    constructor(config = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.sessionDet = null;
        this.sessionRec = null;
        this.dict = [];
        this.recInputType = "float32";
        this.recInputName = "";
        this.isInitialized = false;
    }

    async init() {
        if (this.isInitialized) return;
        await this._setupDebugEnvironment();
        await this._loadDictionary();
        await this._loadModels();
        this.isInitialized = true;
    }

    /**
     * 执行 OCR 识别
     * @param {string|Buffer} imageInput
     * @returns {Promise<{blocks: OcrBlock[], fullText: string}>}
     */
    async recognize(imageInput) {
        if (!this.isInitialized) throw new Error("OCR not initialized");

        const imagePipeline = sharp(imageInput);
        const metadata = await imagePipeline.metadata();

        // 1. 预处理
        const preprocessed = await this._preprocessImage(
            imagePipeline,
            metadata,
        );

        // 2. 检测 (Detection)
        const detOutput = await this._runDetection(preprocessed.tensor);

        // 3. 提取初始 Block
        // 这里返回所有可能的框，不做硬性删除，而是标记状态
        let blocks = this._extractBlocksFromMap(
            detOutput,
            preprocessed.width,
            preprocessed.height,
        );

        if (this.config.saveDebugImages) {
            await this._saveDebugImage(
                preprocessed.buffer,
                preprocessed.width,
                preprocessed.height,
                blocks,
                "01_det_raw",
            );
        }

        // 4. 结构化处理 (去重叠、形状过滤、排序)
        // 注意：这一步会修改 blocks 的顺序，并标记部分 block 为 rejected
        blocks = this._processBlockStructure(blocks);

        // 5. 识别 (Recognition)
        // 仅对未被标记为 rejected 的 block 进行识别
        blocks = await this._runRecognitionPipeline(
            preprocessed.buffer,
            preprocessed.width,
            preprocessed.height,
            blocks,
        );

        // 6. 最终 Debug 图片
        if (this.config.saveDebugImages) {
            await this._saveDebugImage(
                preprocessed.buffer,
                preprocessed.width,
                preprocessed.height,
                blocks,
                "02_final_result",
            );
        }

        // 7. 组装结果
        const validBlocks = blocks.filter((b) => b.status === "valid");
        const fullText = validBlocks.map((b) => b.text).join("\n");

        return {
            blocks, // 包含所有有效和无效的块，方便调试
            validBlocks, // 仅包含有效块
            fullText,
        };
    }

    // =========================================================================
    // 内部逻辑：结构化处理与排序
    // =========================================================================

    /**
     * 处理 Block 的空间关系：过滤形状、移除重叠、排序
     */
    _processBlockStructure(blocks) {
        // 1. 形状过滤 (标记 rejected)
        blocks = this._applyShapeFilter(blocks);

        // 2. 密集重叠去除 (标记 rejected)
        if (this.config.removeDenseClusters) {
            blocks = this._markDenseClusters(blocks);
        }

        // 3. 移除已经是 rejected 的 block (可选：如果不想在后续步骤保留垃圾数据，可以在这里彻底删除)
        // 为了 debug 看得清，我们保留它们，只是不参与后续计算。

        // 4. 排序
        // 只对暂且有效的 block 进行逻辑排序，rejected 的放到最后或保持原样
        const validBlocks = blocks.filter((b) => b.status !== "rejected");
        const rejectedBlocks = blocks.filter((b) => b.status === "rejected");

        const sortedValid = this._sortBlocks(validBlocks);

        // 重新合并，给 valid block 打上序号
        sortedValid.forEach((b, idx) => {
            b.sortIndex = idx;
        });

        return [...sortedValid, ...rejectedBlocks];
    }

    _applyShapeFilter(blocks) {
        if (this.config.filterStrength === "none") return blocks;

        const strength = this.config.filterStrength;
        const params = {
            low: { hRatio: 0.3, wRatio: 1.2 },
            medium: { hRatio: 0.5, wRatio: 2.0 },
            high: { hRatio: 0.8, wRatio: 4.0 },
        }[strength] || { hRatio: 0.5, wRatio: 2.0 };

        // 计算中位数高度
        const validHeights = blocks.map((b) => b.box.h).sort((a, b) => a - b);
        if (validHeights.length === 0) return blocks;
        const medianH = validHeights[Math.floor(validHeights.length / 2)];

        return blocks.map((block) => {
            if (block.status === "rejected") return block;

            const { w, h } = block.box;

            // 绝对尺寸过滤
            if (h < 8 || w < 8) {
                block.status = "rejected";
                block.rejectReason = "too_small";
                return block;
            }
            // 相对高度过滤
            if (h < medianH * params.hRatio) {
                block.status = "rejected";
                block.rejectReason = "height_outlier";
                return block;
            }
            // 细长条过滤
            if (w < h * params.wRatio && h < medianH * 1.5) {
                block.status = "rejected";
                block.rejectReason = "aspect_ratio";
                return block;
            }
            return block;
        });
    }

    _markDenseClusters(blocks) {
        // 仅处理当前有效的 block
        const candidates = blocks.filter((b) => b.status !== "rejected");
        if (candidates.length <= 1) return blocks;

        let maxX = 0, maxY = 0;
        candidates.forEach((b) => {
            maxX = Math.max(maxX, b.box.x + b.box.w);
            maxY = Math.max(maxY, b.box.y + b.box.h);
        });

        const SCALE = 0.25;
        const gridW = Math.ceil(maxX * SCALE) + 1;
        const gridH = Math.ceil(maxY * SCALE) + 1;
        const grid = new Uint8Array(gridW * gridH);

        // Pass 1: Fill Heatmap
        candidates.forEach((b) => {
            const sx = Math.floor(b.box.x * SCALE);
            const ex = Math.floor((b.box.x + b.box.w) * SCALE);
            const sy = Math.floor(b.box.y * SCALE);
            const ey = Math.floor((b.box.y + b.box.h) * SCALE);

            for (let y = sy; y < ey; y++) {
                const rowOffset = y * gridW;
                for (let x = sx; x < ex; x++) {
                    if (x < gridW && y < gridH && grid[rowOffset + x] < 255) {
                        grid[rowOffset + x]++;
                    }
                }
            }
        });

        // Pass 2: Check Overlap
        const clusterLimit = this.config.maxClusterSize;
        const ratioLimit = this.config.overlapRatio;

        candidates.forEach((b) => {
            const sx = Math.floor(b.box.x * SCALE);
            const ex = Math.floor((b.box.x + b.box.w) * SCALE);
            const sy = Math.floor(b.box.y * SCALE);
            const ey = Math.floor((b.box.y + b.box.h) * SCALE);

            let total = 0, dense = 0;
            for (let y = sy; y < ey; y++) {
                const rowOffset = y * gridW;
                for (let x = sx; x < ex; x++) {
                    if (x < gridW && y < gridH) {
                        total++;
                        if (grid[rowOffset + x] >= clusterLimit) dense++;
                    }
                }
            }

            if (total > 0 && (dense / total) > ratioLimit) {
                b.status = "rejected";
                b.rejectReason = "dense_cluster";
            }
        });

        return blocks;
    }

    _sortBlocks(blocks) {
        if (blocks.length === 0) return [];

        // 基础 Y 轴排序
        if (this.config.sortMode === "top-down") {
            return blocks.sort((a, b) => a.box.y - b.box.y);
        }
        if (this.config.sortMode === "raw") {
            return blocks;
        }

        // Column 模式
        // 1. 按 Y 预排序
        const sortedByY = [...blocks].sort((a, b) => a.box.y - b.box.y);
        const columns = [];

        for (const block of sortedByY) {
            let bestColIndex = -1;
            let maxOverlapRatio = 0;
            const maxVerticalGap = block.box.h * 5;

            for (let i = 0; i < columns.length; i++) {
                const col = columns[i];
                const lastBlock = col[col.length - 1];
                const lastBox = lastBlock.box;
                const currBox = block.box;

                // 垂直太远不算一列
                if (currBox.y > lastBox.y + lastBox.h + maxVerticalGap) {
                    continue;
                }

                // 计算水平重叠
                const overlapW = Math.max(
                    0,
                    Math.min(currBox.x + currBox.w, lastBox.x + lastBox.w) -
                        Math.max(currBox.x, lastBox.x),
                );
                const minW = Math.min(currBox.w, lastBox.w);
                const ratio = overlapW / minW;

                if (ratio > 0.4 && ratio > maxOverlapRatio) {
                    maxOverlapRatio = ratio;
                    bestColIndex = i;
                }
            }

            if (bestColIndex !== -1) {
                columns[bestColIndex].push(block);
            } else {
                columns.push([block]);
            }
        }

        // 列间按 X 排序
        columns.sort((colA, colB) => {
            const getCenterX = (col) =>
                col.reduce((sum, b) => sum + (b.box.x + b.box.w / 2), 0) /
                col.length;
            return getCenterX(colA) - getCenterX(colB);
        });

        return columns.flat();
    }

    // =========================================================================
    // 识别流程
    // =========================================================================

    async _runRecognitionPipeline(imageBuffer, w, h, blocks) {
        // 1. 筛选出需要识别的 block
        const activeBlocks = blocks.filter((b) => b.status !== "rejected");
        if (activeBlocks.length === 0) return blocks;

        // 2. 裁剪图片
        const crops = await this._cropBlockImages(
            imageBuffer,
            w,
            h,
            activeBlocks,
        );

        // 3. 按宽度排序以优化 Batch 效率 (记录原始引用关系)
        // Crop 对象结构: { buffer, w, h, blockRef }
        crops.sort((a, b) => a.w - b.w);

        const batchSize = this.config.batchSize;
        for (let i = 0; i < crops.length; i += batchSize) {
            const batch = crops.slice(i, i + batchSize);
            const output = await this._inferRecBatch(batch);

            // 解码并回填到 blockRef 中
            this._decodeBatchOutputToBlocks(output, batch);
        }

        return blocks; // blocks 里的对象已经被原地修改了
    }

    async _cropBlockImages(imageBuffer, w, h, blocks) {
        const recH = this.config.recImgH;
        return Promise.all(blocks.map(async (block) => {
            const box = block.box;
            const recW = Math.round(box.w * (recH / box.h));

            const buffer = await sharp(imageBuffer, {
                raw: { width: w, height: h, channels: 1 },
            })
                .extract({
                    left: box.x,
                    top: box.y,
                    width: box.w,
                    height: box.h,
                })
                .resize(recW, recH, { fit: "fill" })
                .toColourspace("srgb")
                .raw()
                .toBuffer();

            return { buffer, w: recW, h: recH, blockRef: block };
        }));
    }

    _decodeBatchOutputToBlocks(outputTensor, batchItems) {
        const seqLen = outputTensor.dims[1];
        const numClasses = outputTensor.dims[2];
        const data = outputTensor.data;

        for (let b = 0; b < batchItems.length; b++) {
            const item = batchItems[b]; // 包含 blockRef
            const startOffset = b * seqLen * numClasses;
            const itemData = data.subarray(
                startOffset,
                startOffset + seqLen * numClasses,
            );

            // 1. 解码
            const rawText = this._greedyDecode(
                itemData,
                seqLen,
                numClasses,
                item,
            );
            item.blockRef.rawText = rawText;

            // 2. 后处理
            const { text, isValid, reason } = this._postProcessText(rawText);
            item.blockRef.text = text;

            if (!isValid) {
                item.blockRef.status = "rejected";
                item.blockRef.rejectReason = reason;
            } else {
                item.blockRef.status = "valid";
            }
        }
    }

    _postProcessText(text) {
        let result = text.trim();
        let isValid = true;
        let reason = null;

        // 1. Emoji 移除
        if (this.config.removeEmoji) {
            result = result.replace(/\p{Extended_Pictographic}/gu, "").trim();
        }

        // 2. 孤立字符移除
        result = result.replace(/^.\s+/u, "").replace(/\s+.$/u, "");

        // 3. 自定义字符移除
        if (this.config.removeChars.length > 0) {
            for (const char of this.config.removeChars) {
                result = result.split(char).join("");
            }
        }

        // 4. 标点清理
        result = result.replace(/^[\p{P}\p{S}]+/u, "").trim();

        // --- 验证环节 ---

        if (result.length === 0) {
            return { text: "", isValid: false, reason: "empty_after_clean" };
        }

        if (result.length < this.config.textMinLength) {
            return { text: result, isValid: false, reason: "too_short" };
        }

        // 5. 稀疏噪点检测
        if (this.config.trashSparseThresh > 0) {
            const isNoisy = this._checkSparseNoise(result);
            if (isNoisy) {
                return { text: result, isValid: false, reason: "sparse_noise" };
            }
        }

        return { text: result, isValid: true, reason: null };
    }

    _checkSparseNoise(text) {
        const sparseLimit = this.config.trashSparseThresh;
        const tokens = text.split(/\s+/);
        let currentRunCount = 0;
        let totalNoiseLen = 0;
        let currentRunLen = 0;

        for (const token of tokens) {
            if ([...token].length === 1) {
                currentRunLen += token.length + 1; // +1 for space
                currentRunCount++;
            } else {
                if (currentRunCount >= sparseLimit) {
                    totalNoiseLen += currentRunLen;
                }
                currentRunCount = 0;
                currentRunLen = 0;
            }
        }
        if (currentRunCount >= sparseLimit) totalNoiseLen += currentRunLen;

        return totalNoiseLen > text.length * 0.5;
    }

    // =========================================================================
    // 基础辅助函数
    // =========================================================================

    async _setupDebugEnvironment() {
        if (
            this.config.saveDebugImages && !fs.existsSync(this.config.debugDir)
        ) {
            await fs.promises.mkdir(this.config.debugDir, { recursive: true });
        }
    }

    async _loadDictionary() {
        const content = await fs.promises.readFile(
            this.config.dictPath,
            "utf-8",
        );
        this.dict = content.split(/\r?\n/);
    }

    async _loadModels() {
        const detOpt = {
            executionProviders: this.config.executionProviders,
            intraOpNumThreads: this.config.detThreads,
            enableCpuMemArena: true,
        };
        const recOpt = {
            executionProviders: this.config.executionProviders,
            intraOpNumThreads: this.config.recThreads,
            enableCpuMemArena: true,
        };

        [this.sessionDet, this.sessionRec] = await Promise.all([
            ort.InferenceSession.create(this.config.detModel, detOpt),
            ort.InferenceSession.create(this.config.recModel, recOpt),
        ]);

        this.recInputName = this.sessionRec.inputNames[0];
        this.recInputType = this.recInputName.includes("uint8")
            ? "uint8"
            : "float32";
    }

    async _preprocessImage(pipeline, metadata) {
        const { width, height } = this._calculateResizeDimensions(
            metadata.width,
            metadata.height,
        );
        const buffer = await pipeline.clone().resize(width, height).grayscale()
            .raw().toBuffer();
        const tensor = this._imageBufferToFloat32Tensor(buffer, width, height);
        return { tensor, buffer, width, height };
    }

    _calculateResizeDimensions(rawW, rawH) {
        let resizeW, resizeH;
        if (this.config.physicalWidthMM) {
            const widthInch = this.config.physicalWidthMM / 25.4;
            const currentPPI = rawW / widthInch;
            const scale = this.config.targetPPI / currentPPI;
            resizeW = Math.round((rawW * scale) / 32) * 32;
            resizeH = Math.round((rawH * scale) / 32) * 32;
        } else {
            const ratio = Math.min(
                this.config.detLimitSide / Math.max(rawW, rawH),
                1.0,
            );
            resizeW = Math.round(rawW * ratio / 32) * 32;
            resizeH = Math.round(rawH * ratio / 32) * 32;
        }
        return { width: resizeW, height: resizeH };
    }

    _imageBufferToFloat32Tensor(buffer, w, h) {
        const floatArr = new Float32Array(3 * w * h);
        const MAGIC = 0.007843137;
        for (let i = 0; i < w * h; i++) {
            const val = buffer[i] * MAGIC - 1.0;
            floatArr[i] = floatArr[w * h + i] = floatArr[2 * w * h + i] = val;
        }
        return new ort.Tensor("float32", floatArr, [1, 3, h, w]);
    }

    async _runDetection(tensor) {
        const results = await this.sessionDet.run({
            [this.sessionDet.inputNames[0]]: tensor,
        });
        return results[this.sessionDet.outputNames[0]].data;
    }

    _extractBlocksFromMap(predData, w, h) {
        const blocks = [];
        const visited = new Uint8Array(w * h);
        const thresh = this.config.detDbThresh;
        const stride = 2;

        for (let y = 0; y < h; y += stride) {
            for (let x = 0; x < w; x += stride) {
                const idx = y * w + x;
                if (predData[idx] > thresh && visited[idx] === 0) {
                    const box = this._traceComponent(
                        predData,
                        visited,
                        w,
                        h,
                        idx,
                        thresh,
                    );
                    if (box) {
                        blocks.push({
                            box,
                            status: "pending", // 初始状态
                            text: "",
                            rawText: "",
                            sortIndex: -1,
                        });
                    }
                }
            }
        }
        return blocks;
    }

    _traceComponent(predData, visited, w, h, startIdx, thresh) {
        let minX = w, maxX = 0, minY = h, maxY = 0;
        const stack = [startIdx];
        visited[startIdx] = 1;
        let count = 0;

        while (stack.length) {
            const curr = stack.pop();
            const cy = (curr / w) | 0;
            const cx = curr % w;
            count++;

            if (cx < minX) minX = cx;
            if (cx > maxX) maxX = cx;
            if (cy < minY) minY = cy;
            if (cy > maxY) maxY = cy;

            const neighbors = [curr + 1, curr - 1, curr + w, curr - w];
            for (const n of neighbors) {
                if (
                    n >= 0 && n < w * h && visited[n] === 0 &&
                    predData[n] > thresh
                ) {
                    visited[n] = 1;
                    stack.push(n);
                }
            }
        }

        if (count < 10) return null;

        const bw = maxX - minX + 1;
        const bh = maxY - minY + 1;
        const side = Math.min(bw, bh);
        const dilate = side * this.config.unclipRatio / 2.0;
        const pad = this.config.boxPadding;

        return {
            x: Math.max(0, Math.floor(minX - dilate - pad)),
            y: Math.max(0, Math.floor(minY - dilate - pad)),
            w: Math.min(
                w - Math.max(0, Math.floor(minX - dilate - pad)),
                Math.ceil(bw + 2 * dilate + 2 * pad),
            ),
            h: Math.min(
                h - Math.max(0, Math.floor(minY - dilate - pad)),
                Math.ceil(bh + 2 * dilate + 2 * pad),
            ),
        };
    }

    async _inferRecBatch(batchItems) {
        const maxW = batchItems[batchItems.length - 1].w;
        const tensor = this.recInputType === "uint8"
            ? this._createUint8BatchTensor(batchItems, maxW)
            : this._createFloat32BatchTensor(batchItems, maxW);

        const results = await this.sessionRec.run({
            [this.recInputName]: tensor,
        });
        return results[this.sessionRec.outputNames[0]];
    }

    _createUint8BatchTensor(items, maxW) {
        const H = this.config.recImgH;
        const buffer = Buffer.allocUnsafe(items.length * H * maxW * 3);
        let offset = 0;
        const rowBytes = maxW * 3;

        for (const item of items) {
            const srcStride = item.w * 3;
            for (let h = 0; h < H; h++) {
                const srcStart = h * srcStride;
                const dstStart = offset + h * rowBytes;
                item.buffer.copy(
                    buffer,
                    dstStart,
                    srcStart,
                    srcStart + srcStride,
                );
                if (item.w < maxW) {
                    buffer.fill(0, dstStart + srcStride, dstStart + rowBytes);
                }
            }
            offset += H * rowBytes;
        }
        return new ort.Tensor("uint8", buffer, [items.length, H, maxW, 3]);
    }

    _createFloat32BatchTensor(items, maxW) {
        const H = this.config.recImgH;
        const floatArr = new Float32Array(items.length * 3 * H * maxW);
        const planeSize = H * maxW;
        const MAGIC = 0.007843137;

        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            const baseOffset = i * 3 * planeSize;
            let ptr = 0;
            for (let h = 0; h < H; h++) {
                const rRow = baseOffset + (h * maxW);
                const gRow = baseOffset + planeSize + (h * maxW);
                const bRow = baseOffset + 2 * planeSize + (h * maxW);
                for (let w = 0; w < item.w; w++) {
                    const val = item.buffer[ptr++] * MAGIC - 1.0;
                    floatArr[rRow + w] =
                        floatArr[gRow + w] =
                        floatArr[bRow + w] =
                            val;
                }
            }
        }
        return new ort.Tensor("float32", floatArr, [items.length, 3, H, maxW]);
    }

    _greedyDecode(data, seqLen, numClasses, item) {
        let text = "";
        let lastChar = null;
        let blankCount = 0;
        const spaceThresh = this._getSpaceThreshold(item.w, item.h);

        for (let t = 0; t < seqLen; t++) {
            let maxIdx = 0;
            let maxVal = -Infinity;
            const step = t * numClasses;

            for (let c = 0; c < numClasses; c++) {
                if (data[step + c] > maxVal) {
                    maxVal = data[step + c];
                    maxIdx = c;
                }
            }

            if (maxIdx === 0) {
                blankCount++;
                lastChar = null;
                continue;
            }

            const char = this.dict[maxIdx - 1];
            if (!char || char === " " || char === "\r") {
                blankCount++;
                continue;
            }

            if (blankCount >= 1 && text.length > 0) {
                if (
                    this._shouldInsertSpace(
                        text[text.length - 1],
                        char,
                        blankCount,
                        spaceThresh,
                    )
                ) {
                    if (!text.endsWith(" ")) text += " ";
                }
            }

            if (char !== lastChar) text += char;
            lastChar = char;
            blankCount = 0;
        }
        return text;
    }

    _getSpaceThreshold(w, h) {
        if (!this.config.smartSpacing) return this.config.baseBlankThresh;
        const aspectRatio = w / h;
        return aspectRatio < this.config.shortBlockAspectRatio
            ? this.config.shortBlockThresh
            : this.config.baseBlankThresh;
    }

    _shouldInsertSpace(prev, curr, count, thresh) {
        const isChinese = (c) => /[\u4e00-\u9fff]/.test(c);
        const isAlphaNum = (c) => /[a-zA-Z0-9]/.test(c);
        if (isChinese(prev) && isChinese(curr)) return false;
        if (isAlphaNum(prev) && isAlphaNum(curr)) return count >= thresh;
        return true;
    }

    // =========================================================================
    // Debug 绘图增强版
    // =========================================================================

    async _saveDebugImage(buffer, w, h, blocks, suffix) {
        try {
            // 颜色定义
            const COLORS = {
                valid: {
                    stroke: "#00FF00",
                    fill: "rgba(0, 255, 0, 0.1)",
                    text: "#00FF00",
                }, // 绿色：有效
                rejected: {
                    stroke: "#FF0000",
                    fill: "rgba(255, 0, 0, 0.1)",
                    text: "#FF0000",
                }, // 红色：无效
                pending: {
                    stroke: "#0000FF",
                    fill: "rgba(0, 0, 255, 0.1)",
                    text: "#0000FF",
                }, // 蓝色：处理中
            };

            const svgRects = blocks.map((block) => {
                const b = block.box;
                const style = COLORS[block.status] || COLORS.pending;

                // 显示内容：如果有 sortIndex 显示序号，否则显示原因或文本
                let label = "";
                if (block.status === "valid") {
                    // 取前5个字避免遮挡
                    const shortText = block.text.substring(0, 6) +
                        (block.text.length > 6 ? ".." : "");
                    label = `[${block.sortIndex}] ${shortText}`;
                } else {
                    label = block.rejectReason || "REJ";
                }

                // 防止文字越界
                const fontSize = Math.max(8, Math.min(14, b.h / 2));
                const textY = Math.max(fontSize, b.y - 2);

                return `
                    <rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" 
                          fill="${style.fill}" stroke="${style.stroke}" stroke-width="2"/>
                    <text x="${b.x}" y="${textY}" fill="${style.text}" 
                          font-size="${fontSize}" font-family="Arial" font-weight="bold">
                        ${this._escapeXml(label)}
                    </text>
                `;
            }).join("");

            const svg = `<svg width="${w}" height="${h}">${svgRects}</svg>`;
            const outPath = path.join(
                this.config.debugDir,
                `${Date.now()}_${suffix}.png`,
            );

            await sharp(buffer, { raw: { width: w, height: h, channels: 1 } })
                .toColourspace("srgb")
                .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
                .toFile(outPath);

            // console.log(`Debug image saved: ${outPath}`);
        } catch (e) {
            console.error("Debug image save failed:", e);
        }
    }

    _escapeXml(unsafe) {
        return unsafe.replace(/[<>&'"]/g, function (c) {
            switch (c) {
                case "<":
                    return "&lt;";
                case ">":
                    return "&gt;";
                case "&":
                    return "&amp;";
                case "'":
                    return "&apos;";
                case '"':
                    return "&quot;";
            }
        });
    }
}

// -----------------------------------------------------------------------------
// 使用示例
// -----------------------------------------------------------------------------

async function main() {
    const ocr = new Ocr({
        detModel: "./model/det.mobile.onnx",
        recModel: "./model/rec.mobile.nhwc_uint8.onnx",
        dictPath: "./model/dict.txt",

        saveDebugImages: true, // 开启 Debug 图片
        removeDenseClusters: true, // 去除重叠噪点
        sortMode: "column", // 智能分栏排序
        filterStrength: "low",
        textMinLength: 2, // 文本太短会被标红
    });

    try {
        await ocr.init();
        console.time("OCR");
        const result = await ocr.recognize("./sample.png");
        console.timeEnd("OCR");

        console.log("\n------ Final Text (Valid Only) ------");
        console.log(result.fullText);

        console.log("\n------ Block Details (Debug Info) ------");
        result.blocks.forEach((b) => {
            if (b.status === "valid") {
                console.log(
                    `[${b.sortIndex}] Valid: "${b.text}" (box: ${b.box.x},${b.box.y})`,
                );
            } else {
                console.log(
                    `[REJECTED] Reason: ${b.rejectReason}, Raw: "${
                        b.rawText || ""
                    }" (box: ${b.box.x},${b.box.y})`,
                );
            }
        });
    } catch (e) {
        console.error("OCR Error:", e);
    }
}

main();
