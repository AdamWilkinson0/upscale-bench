CXX = clang++
CXXFLAGS = -std=c++17 -O2 -Wall

OBJS = build/main.o build/weights.o build/cpu_conv.o

upscale-bench: $(OBJS)
	$(CXX) $(OBJS) -o $@

build/%.o: src/%.cpp src/*.h
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build upscale-bench
