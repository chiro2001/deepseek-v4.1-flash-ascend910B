# [OPP-OVERRIDE] 起服前把自定义 vendor 复制覆盖到镜像 vendor 路径
if [ -d /opt/dsv41/hcfuse_opp/vendors/custom_transformer ]; then
  _imgv=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
  cp -a /opt/dsv41/hcfuse_opp/vendors/custom_transformer/. "$_imgv/" && echo "[OPP-OVERRIDE] copied -> $_imgv"
  export ASCEND_CUSTOM_OPP_PATH="$_imgv"
fi
