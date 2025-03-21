CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I.
BUILD_DIR = build

LONG_NARDE_SRC = open_spiel/games/long_narde/long_narde.cc
LONG_NARDE_OBJ = $(BUILD_DIR)/$(LONG_NARDE_SRC:.cc=.o)

SIMPLE_TEST_SRC = test_long_narde.cc
SIMPLE_TEST_OBJ = $(BUILD_DIR)/$(SIMPLE_TEST_SRC:.cc=.o)

all: dirs test_long_narde

dirs:
	mkdir -p $(BUILD_DIR)/open_spiel/games/long_narde

$(BUILD_DIR)/%.o: %.cc
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Simple standalone test
test_long_narde: $(SIMPLE_TEST_OBJ) $(LONG_NARDE_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $(BUILD_DIR)/test_long_narde

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all dirs clean 