#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coreml_backend.h"
#include "timer.h"
#include <cstdio>
#include <cstring>

struct CoreMLModel {
    MLModel* model;
};

const char* units_name(int units) {
    switch (units) {
        case UNITS_CPU: return "cpuOnly";
        case UNITS_CPU_GPU: return "cpuAndGPU";
        case UNITS_CPU_NE: return "cpuAndNeuralEngine";
        default: return "all";
    }
}

static MLModelConfiguration* getConfig(int units) {
    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    if (units == UNITS_CPU) cfg.computeUnits = MLComputeUnitsCPUOnly;
    else if (units == UNITS_CPU_GPU) cfg.computeUnits = MLComputeUnitsCPUAndGPU;
    else if (units == UNITS_CPU_NE) cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    else cfg.computeUnits = MLComputeUnitsAll;
    return cfg;
}

std::string coreml_compile(const std::string& mlpackage, double* ms) {
    @autoreleasepool {
        NSError* err = nil;
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mlpackage.c_str()]];
        double t0 = now_ms();
        NSURL* compiled = [MLModel compileModelAtURL:url error:&err];
        if (ms) *ms = now_ms() - t0;
        if (err || !compiled) {
            fprintf(stderr, "could not compile %s\n", mlpackage.c_str());
            return "";
        }
        return compiled.path.UTF8String;
    }
}

CoreMLModel* coreml_load(const std::string& compiled, int units, double* ms) {
    NSError* err = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:compiled.c_str()]];
    double t0 = now_ms();
    MLModel* model = [MLModel modelWithContentsOfURL:url configuration:getConfig(units) error:&err];
    if (ms) *ms = now_ms() - t0;
    if (err || !model) {
        fprintf(stderr, "could not load %s: %s\n", compiled.c_str(), err.localizedDescription.UTF8String);
        return nullptr;
    }
    CoreMLModel* m = new CoreMLModel();
    m->model = model;
    return m;
}

bool coreml_run(CoreMLModel* m, const Tensor& in, Tensor& out) {
    @autoreleasepool {
        NSError* err = nil;
        MLMultiArray* x = [[MLMultiArray alloc] initWithShape:@[@1, @(in.c), @(in.h), @(in.w)]
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&err];
        const float* src = in.data.data();
        size_t bytes = in.data.size() * sizeof(float);
        [x getMutableBytesWithHandler:^(void* ptr, NSInteger size, NSArray<NSNumber*>* strides) {
            memcpy(ptr, src, bytes);
        }];

        MLDictionaryFeatureProvider* feats = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x" : x} error:&err];
        id<MLFeatureProvider> res = [m->model predictionFromFeatures:feats error:&err];
        if (err || !res) {
            fprintf(stderr, "prediction failed: %s\n", err.localizedDescription.UTF8String);
            return false;
        }

        MLMultiArray* y = [res featureValueForName:@"y"].multiArrayValue;
        out = Tensor(y.shape[1].intValue, y.shape[2].intValue, y.shape[3].intValue);
        float* dst = out.data.data();
        size_t count = out.data.size();
        // output comes back as float16 sometimes even when asking for float32
        bool isHalf = (y.dataType == MLMultiArrayDataTypeFloat16);
        [y getBytesWithHandler:^(const void* ptr, NSInteger size) {
            if (isHalf) {
                const _Float16* p = (const _Float16*)ptr;
                for (size_t i=0; i<count; i++) dst[i] = (float)p[i];
            } else {
                memcpy(dst, ptr, count * sizeof(float));
            }
        }];
        return true;
    }
}

void coreml_destroy(CoreMLModel* m) {
    delete m;
}
