/**
 * nvdsparsebbox_sam3.cpp  (SAM3Fixed/)
 *
 * Custom DeepStream output parsers for SAM3Fixed/ (baked-classes) engines:
 *
 *   --mask 0:  detections [N, 6]  float32   x1 y1 x2 y2 score cls_id
 *              → NvDsInferParseSAM3Det
 *
 *   --mask 1:  detections [N, 6]  float32
 *              masks      [N, 36, 36]  float16  sigmoid already applied
 *              → NvDsInferParseSAM3Full
 *
 * N = num_classes * Q  (Q=200)   cls_id for query i: i / Q
 *
 * Build (from this directory):
 *   g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
 *       -I/opt/nvidia/deepstream/deepstream/sources/includes \
 *       -I/usr/local/cuda-13.0/targets/x86_64-linux/include \
 *       -std=c++14
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static const int   Q               = 200;
static const float SCORE_THRESHOLD = 0.3f;
static const float NMS_IOU_THRESH  = 0.6f;
static const int   MASK_H          = 36;
static const int   MASK_W          = 36;

// ── FP16 → FP32 (portable, no CUDA headers needed) ───────────────────────────

static float fp16_to_float(uint16_t h)
{
    uint32_t sign =  (h >> 15) & 0x1u;
    uint32_t exp  =  (h >> 10) & 0x1fu;
    uint32_t mant =   h        & 0x3ffu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// ── IoU ───────────────────────────────────────────────────────────────────────

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

// ── shared detection struct ───────────────────────────────────────────────────

struct Det {
    float x1, y1, x2, y2, score;
    int   cls_id, query_idx;
};

static std::vector<Det> collect_and_nms(
    const uint16_t *det_data, int N,
    const std::vector<float> &threshVec)
{
    std::vector<Det> candidates;
    candidates.reserve(32);

    for (int i = 0; i < N; ++i) {
        const uint16_t *d = det_data + i * 6;
        float score  = fp16_to_float(d[4]);
        int   cls    = (int)fp16_to_float(d[5]);

        float thresh = SCORE_THRESHOLD;
        if (!threshVec.empty())
            thresh = (cls >= 0 && cls < (int)threshVec.size())
                     ? threshVec[cls] : threshVec[0];

        if (score < thresh) continue;
        float x1 = fp16_to_float(d[0]), y1 = fp16_to_float(d[1]);
        float x2 = fp16_to_float(d[2]), y2 = fp16_to_float(d[3]);
        if (x2 <= x1 || y2 <= y1) continue;

        candidates.push_back({x1, y1, x2, y2, score, cls, i});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Det &a, const Det &b){ return a.score > b.score; });

    std::vector<bool> sup(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (sup[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (sup[j] || candidates[i].cls_id != candidates[j].cls_id) continue;
            if (iou_xyxy(candidates[i].x1, candidates[i].y1,
                         candidates[i].x2, candidates[i].y2,
                         candidates[j].x1, candidates[j].y1,
                         candidates[j].x2, candidates[j].y2) > NMS_IOU_THRESH)
                sup[j] = true;
        }
    }

    std::vector<Det> out;
    for (size_t i = 0; i < candidates.size(); ++i)
        if (!sup[i]) out.push_back(candidates[i]);
    return out;
}

// ── NvDsInferParseSAM3Det  (--mask 0: detections only) ───────────────────────

extern "C"
bool NvDsInferParseSAM3Det(
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

    const uint16_t *det_data = (const uint16_t *)det_layer->buffer;
    auto dets = collect_and_nms(det_data, N, detectionParams.perClassPreclusterThreshold);

    for (const auto &d : dets) {
        NvDsInferObjectDetectionInfo obj = {};
        obj.classId             = (unsigned int)d.cls_id;
        obj.detectionConfidence = d.score;
        obj.left                = d.x1;
        obj.top                 = d.y1;
        obj.width               = d.x2 - d.x1;
        obj.height              = d.y2 - d.y1;
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseSAM3Det);

// ── NvDsInferParseSAM3Full  (--mask 1: detections + masks fp16) ──────────────

extern "C"
bool NvDsInferParseSAM3Full(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo              const &networkInfo,
    NvDsInferParseDetectionParams     const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>  &objectList)
{
    const NvDsInferLayerInfo *det_layer   = nullptr;
    const NvDsInferLayerInfo *masks_layer = nullptr;

    for (const auto &l : outputLayersInfo) {
        std::string name(l.layerName);
        if (name == "detections") det_layer   = &l;
        if (name == "masks")      masks_layer = &l;
    }
    if (!det_layer || !masks_layer) {
        if (outputLayersInfo.size() >= 2) {
            det_layer   = &outputLayersInfo[0];
            masks_layer = &outputLayersInfo[1];
        } else {
            return false;
        }
    }

    int N = det_layer->inferDims.d[0];
    if (N <= 0) return false;

    int mask_h = (masks_layer->inferDims.numDims >= 3) ? masks_layer->inferDims.d[1] : MASK_H;
    int mask_w = (masks_layer->inferDims.numDims >= 3) ? masks_layer->inferDims.d[2] : MASK_W;

    const uint16_t *det_data   = (const uint16_t *)det_layer->buffer;
    const uint16_t *masks_fp16 = (const uint16_t *)masks_layer->buffer;

    float net_w = (float)networkInfo.width;
    float net_h = (float)networkInfo.height;

    auto dets = collect_and_nms(det_data, N, detectionParams.perClassPreclusterThreshold);

    for (const auto &d : dets) {
        int cx1 = std::max(0,      (int)( d.x1 * mask_w / net_w));
        int cy1 = std::max(0,      (int)( d.y1 * mask_h / net_h));
        int cx2 = std::min(mask_w, (int)std::ceil(d.x2 * mask_w / net_w));
        int cy2 = std::min(mask_h, (int)std::ceil(d.y2 * mask_h / net_h));
        int cw  = cx2 - cx1;
        int ch  = cy2 - cy1;

        NvDsInferInstanceMaskInfo obj = {};
        obj.classId             = (unsigned int)d.cls_id;
        obj.detectionConfidence = d.score;
        obj.left                = d.x1;
        obj.top                 = d.y1;
        obj.width               = d.x2 - d.x1;
        obj.height              = d.y2 - d.y1;

        if (cw > 0 && ch > 0) {
            float *mask_crop = (float *)std::malloc(cw * ch * sizeof(float));
            if (mask_crop) {
                const uint16_t *src = masks_fp16 + d.query_idx * mask_h * mask_w;
                for (int r = 0; r < ch; ++r) {
                    const uint16_t *row_src = src + (cy1 + r) * mask_w + cx1;
                    float          *row_dst = mask_crop + r * cw;
                    for (int c = 0; c < cw; ++c)
                        row_dst[c] = fp16_to_float(row_src[c]);
                }
                obj.mask        = mask_crop;
                obj.mask_width  = (unsigned int)cw;
                obj.mask_height = (unsigned int)ch;
                obj.mask_size   = cw * ch * sizeof(float);
            }
        }
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseSAM3Full);
