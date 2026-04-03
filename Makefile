#---------------------------------------------------------------------
# Makefile for Rotor (with destructive but safe clean)
#
# Author : Jean-Luc PONS (base)
# Patch  : destructive clean with safety checks
#---------------------------------------------------------------------

SRCDIR = Rotor-Cuda
OBJDIR = obj
TARGET = Rotor

SRC = $(SRCDIR)/Base58.cpp $(SRCDIR)/IntGroup.cpp $(SRCDIR)/Main.cpp $(SRCDIR)/MaskedSearch.cpp \
      $(SRCDIR)/Bloom.cpp $(SRCDIR)/Random.cpp $(SRCDIR)/Sort.cpp $(SRCDIR)/Timer.cpp \
      $(SRCDIR)/Int.cpp $(SRCDIR)/IntMod.cpp $(SRCDIR)/Point.cpp $(SRCDIR)/SECP256K1.cpp \
      $(SRCDIR)/Rotor.cpp $(SRCDIR)/GPU/GPUGenerate.cpp $(SRCDIR)/hash/ripemd160.cpp \
      $(SRCDIR)/hash/sha256.cpp $(SRCDIR)/hash/sha512.cpp $(SRCDIR)/hash/ripemd160_sse.cpp \
      $(SRCDIR)/hash/sha256_sse.cpp $(SRCDIR)/hash/keccak160.cpp $(SRCDIR)/GmpUtil.cpp \
      $(SRCDIR)/CmdParse.cpp

ifdef gpu

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o Main.o MaskedSearch.o Bloom.o Random.o Sort.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Rotor.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o hash/keccak160.o \
        GPU/GPUEngine.o GPU/MaskedGPUEngine.o GmpUtil.o CmdParse.o)

else

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o Main.o MaskedSearch.o Bloom.o Random.o Sort.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Rotor.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o hash/keccak160.o \
        GmpUtil.o CmdParse.o)

endif

CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc
# nvcc requires joint notation w/o dot, i.e. "7.5" -> "75"
ccap       = $(shell echo $(CCAP) | tr -d '.')

ifdef gpu
ifdef debug
CXXFLAGS   = -DWITHGPU -m64  -mssse3 -Wno-write-strings -g -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include
else
CXXFLAGS   =  -DWITHGPU -m64 -mssse3 -Wno-write-strings -O2 -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include
endif
LFLAGS     = -lgmp -lpthread -L$(CUDA)/lib64 -lcudart
else
ifdef debug
CXXFLAGS   = -m64 -mssse3 -Wno-write-strings -g -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include
else
CXXFLAGS   =  -m64 -mssse3 -Wno-write-strings -O2 -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include
endif
LFLAGS     = -lgmp -lpthread
endif

#--------------------------------------------------------------------
.PHONY: all clean distclean veryclean

ifdef gpu
ifdef debug
$(OBJDIR)/GPU/GPUEngine.o: $(SRCDIR)/GPU/GPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c $(SRCDIR)/GPU/GPUEngine.cu

$(OBJDIR)/GPU/MaskedGPUEngine.o: $(SRCDIR)/GPU/MaskedGPUEngine.cu
	$(NVCC) -G -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -g -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/MaskedGPUEngine.o -c $(SRCDIR)/GPU/MaskedGPUEngine.cu
else
$(OBJDIR)/GPU/GPUEngine.o: $(SRCDIR)/GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O2 -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c $(SRCDIR)/GPU/GPUEngine.cu

$(OBJDIR)/GPU/MaskedGPUEngine.o: $(SRCDIR)/GPU/MaskedGPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O3 -I$(SRCDIR) -I$(SRCDIR)/GPU -I$(SRCDIR)/hash -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/MaskedGPUEngine.o -c $(SRCDIR)/GPU/MaskedGPUEngine.cu
endif
endif

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: $(TARGET)

$(TARGET): $(OBJET)
	@echo Making Rotor...
	$(CXX) $(OBJET) $(LFLAGS) -o $(TARGET)

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p hash

# -------------------------------------------------------------------
# Destructive clean: delete obj dir completely and remove Rotor binary
# with safety checks to avoid accidental rm -rf /
clean:
	@echo "Running destructive clean: deleting $(OBJDIR) and $(TARGET) (if present)..."
	@sh -c '\
		if [ -z "$(OBJDIR)" ] || [ "$(OBJDIR)" = "/" ] || [ "$(OBJDIR)" = "." ]; then \
			echo "ERROR: unsafe OBJDIR=$(OBJDIR) — aborting clean"; exit 1; \
		fi; \
		# If OBJDIR exists, remove it entirely (recursively) \
		if [ -d "$(OBJDIR)" ]; then \
			echo "Removing directory: $(OBJDIR)"; rm -rf "$(OBJDIR)"; \
		else \
			echo "OBJDIR $(OBJDIR) does not exist, skipping rm -rf"; \
		fi; \
		# Remove binary if present \
		if [ -f "$(TARGET)" ]; then echo "Removing : $(TARGET)"; rm -f "$(TARGET)"; else echo "No $(TARGET) found"; fi; \
		# Recreate minimal obj tree for next build \
		mkdir -p "$(OBJDIR)"; mkdir -p "$(OBJDIR)"/GPU; mkdir -p "$(OBJDIR)"/hash; \
		echo "Destructive clean complete."; \
	'

# distclean removes also temporary directories (keeps safe check)
distclean: clean
	@echo "Distclean: ensuring $(OBJDIR) is fully reset..."
	@sh -c '\
		if [ -z "$(OBJDIR)" ] || [ "$(OBJDIR)" = "/" ] || [ "$(OBJDIR)" = "." ]; then \
			echo "Refusing to distclean: unsafe OBJDIR=$(OBJDIR)"; exit 1; \
		fi; \
		rm -rf "$(OBJDIR)"; \
		mkdir -p "$(OBJDIR)"/GPU "$(OBJDIR)"/hash; \
		echo "Distclean complete."; \
	'

veryclean: distclean
	@echo "Very clean: removing also binary"
	@rm -f $(TARGET)

# -------------------------------------------------------------------
