import sys
def link_accelerate():
    assert sys.platform == "darwin"
    return True, ["-DRL_TOOLS_BACKEND_ENABLE_ACCELERATE", "-framework", "Accelerate"]