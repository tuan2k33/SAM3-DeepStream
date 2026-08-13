/**
 * nvdsparsebbox_sam3.cpp  (SAM3Open/)
 *
 * Custom DeepStream plugin for SAM3Open/ (dynamic-prompt) vision_decoder.engine:
 *
 *   - NvDsInferInitializeInputLayers : injects text_features [32,256] float32
 *     + text_mask [32] uint8 (TensorRT reports this as BOOL, DeepStream maps
 *     it to UINT8 -- verified empirically) as non-image input layers, one row
 *     per camera slot, read from two flat snapshot files written by
 *     sam3open_text.py (same directory):
 *         /dev/shm/sam3open_text_features.bin   MAX_SOURCES x 32 x 256  f32
 *         /dev/shm/sam3open_text_mask.bin       MAX_SOURCES x 32        u8
 *     Called once per nvinfer context (re)load -- reloading the SAME
 *     config-file-path via nvinfer.set() re-triggers it without a pipeline
 *     restart or engine rebuild, verified against the real engine. nvinfer
 *     looks this up by this EXACT symbol name (dlsym); a custom name is
 *     silently never called.
 *
 *   - NvDsInferParseSam3OpenDet : parses `detections` [200,5] float32
 *     (x1 y1 x2 y2 score, pixel units, already scaled -- see
 *     DynamicVisionDecoderWrapper in export.py). Only one prompt per camera
 *     row, so there's no cls_id column -- classId is always 0; Python sets
 *     the display label from the camera's own current prompt string.
 *
 * Build (from this directory):
 *   g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
 *       -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *       -I/usr/local/cuda-13.0/targets/x86_64-linux/include \
 *       -std=c++14
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static const float NMS_IOU_THRESH = 0.6f;

static float iou_xyxy(float ax1, float ay1, float ax2, float ay2,
                      float bx1, float by1, float bx2, float by2)
{
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    if (inter == 0.0f) return 0.0f;
    float a = (ax2 - ax1) * (ay2 - ay1);
    float b = (bx2 - bx1) * (by2 - by1);
    return inter / (a + b - inter);
}

struct Det { float x1, y1, x2, y2, score; };

static size_t elemSize(NvDsInferDataType t)
{
    switch (t) {
    case INT64: return 8;
    case INT32: case FLOAT: return 4;
    case HALF: return 2;
    case INT8: case UINT8: return 1;
    default: return 0;
    }
}

extern "C" bool NvDsInferInitializeInputLayers(
    std::vector<NvDsInferLayerInfo> const &inputLayers,
    NvDsInferNetworkInfo const &networkInfo,
    unsigned int maxBatchSize)
{
    for (auto const &layer : inputLayers) {
        size_t es = elemSize(layer.dataType);
        size_t rowBytes = layer.inferDims.numElements * es;
        size_t totalBytes = rowBytes * maxBatchSize;
        std::string path = std::string("/dev/shm/sam3open_") + layer.layerName + ".bin";

        FILE *f = fopen(path.c_str(), "rb");
        if (!f) {
            printf("[SAM3Open] MISSING snapshot %s (write it before pipeline.start())\n", path.c_str());
            return false;
        }
        size_t rd = fread(layer.buffer, 1, totalBytes, f);
        fclose(f);
        if (rd != totalBytes) {
            printf("[SAM3Open] SHORT read %s: %zu/%zu bytes (maxBatchSize=%u)\n",
                   path.c_str(), rd, totalBytes, maxBatchSize);
            return false;
        }
        printf("[SAM3Open] loaded %s: %u rows x %zu bytes\n", layer.layerName, maxBatchSize, rowBytes);
    }
    fflush(stdout);
    return true;
}

extern "C"
bool NvDsInferParseSam3OpenDet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo              const &networkInfo,
    NvDsInferParseDetectionParams     const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    const NvDsInferLayerInfo *det_layer = nullptr;
    for (const auto &l : outputLayersInfo)
        if (std::string(l.layerName) == "detections") { det_layer = &l; break; }
    if (!det_layer && !outputLayersInfo.empty())
        det_layer = &outputLayersInfo[0];
    if (!det_layer) return false;

    int N = det_layer->inferDims.d[0];
    if (N <= 0) return false;

    float thresh = detectionParams.perClassPreclusterThreshold.empty()
                   ? 0.3f : detectionParams.perClassPreclusterThreshold[0];

    const float *d = (const float *)det_layer->buffer;
    std::vector<Det> candidates;
    candidates.reserve(16);
    for (int i = 0; i < N; ++i) {
        const float *row = d + i * 5;
        float score = row[4];
        if (score < thresh) continue;
        float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
        if (x2 <= x1 || y2 <= y1) continue;
        candidates.push_back({x1, y1, x2, y2, score});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Det &a, const Det &b){ return a.score > b.score; });

    std::vector<bool> sup(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (sup[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (sup[j]) continue;
            if (iou_xyxy(candidates[i].x1, candidates[i].y1, candidates[i].x2, candidates[i].y2,
                         candidates[j].x1, candidates[j].y1, candidates[j].x2, candidates[j].y2) > NMS_IOU_THRESH)
                sup[j] = true;
        }
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (sup[i]) continue;
        NvDsInferObjectDetectionInfo obj = {};
        obj.classId             = 0;
        obj.detectionConfidence = candidates[i].score;
        obj.left                = candidates[i].x1;
        obj.top                 = candidates[i].y1;
        obj.width               = candidates[i].x2 - candidates[i].x1;
        obj.height              = candidates[i].y2 - candidates[i].y1;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseSam3OpenDet);
