# ——— ANSI color & style codes ———
RESET = "\033[0m"

# regular
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
ORANGE = "\033[38;2;255;165;0m"
BROWN = "\033[38;2;165;105;55m"
VIOLET = "\033[38;2;138;43;226m"
TURQUOISE = "\033[38;2;64;224;208m"

# bright/high-intensity
BRIGHT_BLACK = "\033[90m"
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"
BRIGHT_ORANGE = "\033[38;2;255;200;0m"
LIGHT_GRAY = "\033[38;5;252m"

# text styles
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
REVERSED = "\033[7m"


# LLM answer block colors
ANSWER_BG = "\033[48;5;238m"  # dark gray background (256-color)
ANSWER_FG = "\033[38;5;252m"  # light gray text (256-color)

# Visual-marker highlight colours (mark_text=True)
MARKED_DOCS_HIGHLIGHT_COLOR = "yellow"  # relevant source chunks
MARKED_DOCS_ANSWER_MARK_COLOR = ""  # grounded/effective matches (HTML/MD) — empty = plain <mark> (OpenWebUI DOMPurify strips style= attributes)
MARKED_DOCS_ANSWER_ANSI_COLOR = "48;5;214"  # grounded/effective matches (CLI terminal)

# (RESET is defined at line 2)
