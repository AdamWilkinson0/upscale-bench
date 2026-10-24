#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "mps_backend.h"

struct MPSNet {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    MPSGraphDevice* graphDevice;
    MPSGraph* graph;
    MPSGraphTensor* input;
    MPSGraphTensor* output;
};

MPSNet* mps_create(const Weights& net) {
    MPSNet* m = new MPSNet();
    m->device = MTLCreateSystemDefaultDevice();
    m->queue = [m->device newCommandQueue];
    m->graphDevice = [MPSGraphDevice deviceWithMTLDevice:m->device];
    m->graph = [[MPSGraph alloc] init];

    MPSGraph* g = m->graph;
    m->input = [g placeholderWithShape:@[@1, @(net.layers[0].in_c), @-1, @-1] dataType:MPSDataTypeFloat32 name:@"x"];
    MPSGraphTensor* x = m->input;

    for (size_t i = 0; i < net.layers.size(); i++) {
        const ConvLayer &l = net.layers[i];
        NSData* wData = [NSData dataWithBytes:l.w.data() length:l.w.size() * sizeof(float)];
        NSData* bData = [NSData dataWithBytes:l.b.data() length:l.b.size() * sizeof(float)];
        MPSGraphTensor* w = [g constantWithData:wData shape:@[@(l.out_c), @(l.in_c), @(l.k), @(l.k)] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* b = [g constantWithData:bData shape:@[@1, @(l.out_c), @1, @1] dataType:MPSDataTypeFloat32];

        int pad = l.k / 2;
        MPSGraphConvolution2DOpDescriptor* desc =
            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:1 strideInY:1
                                                       dilationRateInX:1 dilationRateInY:1
                                                                groups:1
                                                           paddingLeft:pad paddingRight:pad
                                                            paddingTop:pad paddingBottom:pad
                                                          paddingStyle:MPSGraphPaddingStyleExplicit
                                                            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                         weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        x = [g convolution2DWithSourceTensor:x weightsTensor:w descriptor:desc name:nil];
        x = [g additionWithPrimaryTensor:x secondaryTensor:b name:nil];
        if (i != net.layers.size() - 1) x = [g reLUWithTensor:x name:nil];
    }

    m->output = [g depthToSpace2DTensor:x widthAxis:3 heightAxis:2 depthAxis:1
                              blockSize:net.scale usePixelShuffleOrder:YES name:@"y"];
    return m;
}

Tensor mps_run(MPSNet* m, const Tensor& in) {
    @autoreleasepool {
        NSData* data = [NSData dataWithBytesNoCopy:(void*)in.data.data()
                                            length:in.data.size() * sizeof(float)
                                      freeWhenDone:NO];
        MPSGraphTensorData* feed = [[MPSGraphTensorData alloc] initWithDevice:m->graphDevice
                                                                         data:data
                                                                        shape:@[@1, @(in.c), @(in.h), @(in.w)]
                                                                     dataType:MPSDataTypeFloat32];
        // NSDictionary* res = [m->graph runWithFeeds:@{m->input : feed} targetTensors:@[m->output] targetOperations:nil];
        // ^ gets slower every frame at 1080p, use our own queue
        NSDictionary* res = [m->graph runWithMTLCommandQueue:m->queue feeds:@{m->input : feed} targetTensors:@[m->output] targetOperations:nil];
        MPSGraphTensorData* y = res[m->output];
        NSArray<NSNumber*>* shape = y.shape;
        Tensor out(shape[1].intValue, shape[2].intValue, shape[3].intValue);
        [[y mpsndarray] readBytes:out.data.data() strideBytes:nil];
        return out;
    }
}

std::string mps_device_name(MPSNet* m) {
    return m->device.name.UTF8String;
}

void mps_destroy(MPSNet* m) {
    delete m;
}
