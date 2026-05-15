CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall
SRCDIR   = src
BINDIR   = bin
GENDIR   = $(SRCDIR)/generator

ALGOS    = proposed md lb uc aga local_only nearest_only all_global no_drop
TARGETS  = $(addprefix $(BINDIR)/, $(ALGOS))

.PHONY: all clean

all: $(TARGETS) $(BINDIR)/gen_data

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/%: $(SRCDIR)/%.cpp $(SRCDIR)/common.h | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BINDIR)/gen_data: $(GENDIR)/gen_data.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -rf $(BINDIR)
