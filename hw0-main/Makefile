PYTHON_LIB_DIR := D:/A-sevenTen/Anaconda/Anaconda_install/envs/needle
PYTHON_LIB := libpython310.a

# 使用虚拟环境中的 Python 解释器
PYTHON := python

# 获取 Python 和 Pybind11 的包含路径
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_INC := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")

# 编译器和编译标志
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++11 -fPIC $(PYBIND11_INC) -I$(PYTHON_INC)

# 目标和源文件
TARGET := src/simple_ml_ext.pyd
SRCS := src/simple_ml_ext.cpp

# 默认目标
all: $(TARGET)

# 链接共享库
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -L$(PYTHON_LIB_DIR) -lpython310

# 清理编译生成的文件
clean:
	del /f /q $(TARGET)

.PHONY: all clean