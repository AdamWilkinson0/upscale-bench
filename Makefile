CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall
OBJCFLAGS = $(CXXFLAGS) -fobjc-arc
FRAMEWORKS = -framework Foundation -framework Metal -framework MetalPerformanceShaders \
             -framework MetalPerformanceShadersGraph -framework CoreML -framework Accelerate

OBJS = build/main.o build/weights.o build/cpu_conv.o build/cpu_fast.o build/image.o \
       build/mps_backend.o build/coreml_backend.o

upscale-bench: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(FRAMEWORKS)

build/%.o: src/%.cpp src/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -c $< -o $@

build/image.o: src/image.cpp src/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -w -c $< -o $@

build/%.o: src/%.mm src/*.h
	@mkdir -p build
	$(CXX) $(OBJCFLAGS) -c $< -o $@

clean:
	rm -rf build upscale-bench
