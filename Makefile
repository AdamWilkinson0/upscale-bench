CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall
FRAMEWORKS = -framework Accelerate

OBJS = build/main.o build/weights.o build/cpu_conv.o build/cpu_fast.o

upscale-bench: $(OBJS)
	$(CXX) $(OBJS) -o $@ $(FRAMEWORKS)

build/%.o: src/%.cpp src/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -DACCELERATE_NEW_LAPACK -c $< -o $@

clean:
	rm -rf build upscale-bench
