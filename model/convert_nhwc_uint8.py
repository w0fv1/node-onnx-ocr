# ./model/convert_nhwc_uint8.py
import sys
import onnx # type: ignore
from onnx import helper, TensorProto # type: ignore

# =================配置区域=================
# 输出 Rec 模型：FP32 推理 + Uint8 NHWC 输入 + 内置预处理
OUTPUT_REC = "./rec.server.nhwc_uint8.onnx"
# ==========================================


def update_rec_model_input_to_nhwc_uint8(input_path: str = "rec.onnx") -> None:
    """
    将 Rec 模型的输入改为 Uint8 NHWC，并在模型内部插入预处理：
    Uint8[N, H, W, 3] -> Cast(FP32) -> 归一化 -> Transpose(NCHW) -> 原模型输入。
    不做权重量化，仅改输入类型和预处理逻辑。
    """
    print(f"[Rec] 开始修改输入层，输入模型路径：{input_path}")

    # 1. 加载原模型
    model = onnx.load(input_path)
    graph = model.graph

    # 获取原始输入节点名称
    original_input = graph.input[0]
    orig_input_name = original_input.name

    # 2. 定义新的 Uint8 NHWC 输入节点
    new_input_name = "image_nhwc_uint8"
    new_input = helper.make_tensor_value_info(
        new_input_name,
        TensorProto.UINT8,
        ["batch", "height", "width", 3],
    )

    # 3. 构建预处理节点链

    # A. Cast: Uint8 -> Float
    cast_output = "pre_cast_out"
    node_cast = helper.make_node(
        "Cast",
        inputs=[new_input_name],
        outputs=[cast_output],
        to=TensorProto.FLOAT,
    )

    # B. Normalize: (x * 1/127.5) - 1.0
    # Mul
    mul_val_name = "pre_mul_val"
    mul_output = "pre_mul_out"
    # 1/127.5 ≈ 0.007843137
    mul_tensor = helper.make_tensor(
        mul_val_name,
        TensorProto.FLOAT,
        [1],
        [0.007843137],
    )
    node_mul = helper.make_node(
        "Mul",
        inputs=[cast_output, mul_val_name],
        outputs=[mul_output],
    )

    # Sub
    sub_val_name = "pre_sub_val"
    sub_output = "pre_sub_out"
    sub_tensor = helper.make_tensor(
        sub_val_name,
        TensorProto.FLOAT,
        [1],
        [1.0],
    )
    node_sub = helper.make_node(
        "Sub",
        inputs=[mul_output, sub_val_name],
        outputs=[sub_output],
    )

    # C. Transpose: NHWC -> NCHW
    # 输入: [N, H, W, C] -> 输出: [N, C, H, W]
    node_transpose = helper.make_node(
        "Transpose",
        inputs=[sub_output],
        outputs=[orig_input_name],
        perm=[0, 3, 1, 2],
    )

    # 4. 组装 Graph：在现有图前面插入预处理节点
    graph.node.insert(0, node_transpose)
    graph.node.insert(0, node_sub)
    graph.node.insert(0, node_mul)
    graph.node.insert(0, node_cast)

    # 添加常量参数
    graph.initializer.extend([mul_tensor, sub_tensor])

    # 替换输入定义为新的 Uint8 NHWC 输入
    graph.input.remove(original_input)
    graph.input.insert(0, new_input)

    # 5. 保存模型
    onnx.save(model, OUTPUT_REC)
    print(f"[Rec] 修改完成，输出模型路径：{OUTPUT_REC}")


if __name__ == "__main__":
    # 命令行参数：python modify_rec_input.py [input_onnx_path]
    if len(sys.argv) > 1:
        input_rec_path = sys.argv[1]
    else:
        input_rec_path = "rec.onnx"

    update_rec_model_input_to_nhwc_uint8(input_rec_path)
    print("[Rec] 模型处理流程结束")
