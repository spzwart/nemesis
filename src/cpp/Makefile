PYTHON_CONFIG := python3.12-config

CXX = g++
CXXFLAGS = -std=c++11 -O2 -fPIC $(shell $(PYTHON_CONFIG) --includes) -fopenmp
LDFLAGS = -shared $(shell $(PYTHON_CONFIG) --ldflags) -fopenmp

SRC_DIR = .
SRC = $(wildcard $(SRC_DIR)/*.cpp)
BUILD_DIR = ../build/

OUT = $(patsubst $(SRC_DIR)%.cpp, $(BUILD_DIR)/%_worker.so, $(SRC))

all: $(OUT)

$(BUILD_DIR)/%_worker.so: $(SRC_DIR)%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(OUT)
