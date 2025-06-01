import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    A Triton “python” backend that accepts two INT32 tensors:
      • input_ids        (shape [batch, seq_len])
      • attention_mask   (shape [batch, seq_len])
    It forwards them to the corresponding sub‐model (which Triton has already
    loaded as either a .plan or a .onnx backend), then returns the raw logits.
    """

    def initialize(self, args):
        # Triton sets these keys for us:
        #   args["model_repository"]  → path to triton_repo/
        #   args["model_instance_kind"]  → "GPU" or "CPU"
        #   args["model_instance_device_id"] → the GPU index (e.g. 0)
        #
        # We also expect two parameters in config.pbtxt: "MODEL_NAME" and "FOLD_IDX"
        params = args.get("parameters", {})
        self.submodel = params["MODEL_NAME"]["string_value"]
        self.fold = int(params["FOLD_IDX"]["string_value"])
        # e.g. self.submodel == "model0", self.fold == 0
        # But the actual Triton model name for this sub‐model will be
        #   f"{self.submodel}_fold{self.fold}_stage2"
        self.full_name = f"model{self.submodel}_fold{self.fold}_stage2"

        # We’ll cache the “model_name” to which we forward every request.
        # Triton ensures that “self.full_name” is already loaded (either as
        # tensorrt_plan or onnxruntime_onnx) in the same Triton server.
        self.triton_client = pb_utils.TritonClient()

    def execute(self, requests):
        responses = []
        for request in requests:
            # 1) Read input tensors from Triton
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
            attention_mask = pb_utils.get_input_tensor_by_name(
                request, "attention_mask"
            ).as_numpy()

            # 2) Forward to the sub‐model (TensorRT plan or ONNX) inside Triton
            infer_request = pb_utils.InferenceRequest(
                model_name=self.full_name,
                requested_output_names=["logits"],
                inputs=[
                    pb_utils.Tensor("input_ids", input_ids),
                    pb_utils.Tensor("attention_mask", attention_mask),
                ],
            )
            infer_response = infer_request.exec()

            # 3) Grab the “logits” output
            logits_tensor = pb_utils.get_output_tensor_by_name(infer_response, "logits")
            logits = logits_tensor.as_numpy()  # shape (batch, 1) or (batch,)

            # 4) Return that same logits as our Python‐backend’s output
            out_tensor = pb_utils.Tensor("logits", logits)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
