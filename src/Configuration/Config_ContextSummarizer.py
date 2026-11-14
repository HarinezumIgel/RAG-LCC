"""
Configuration for ContextSummarizer
"""

# Context summarizer specific configuration values
# These are defined here to be separate from global configuration

# Chunk overlap configuration (number of chunks to overlap)
CHUNK_OVERLAP = 5
SUMMARY_LENGTH = "medium"  # Options: short, medium, long, very_long

# {max_words} is resolved from SUMMARY_WORD_LIMITS in Config_Grounder.
SUMMARY_PROMPT_TEMPLATE = (
    "Please summarize the following text in at most {max_words} words. Do not pad:"
)
_PROMPT_FINAL_SUMMARY_PROMPT_TEMPLATE = (
    "Combine the following partial summaries into one comprehensive document summary. "
    "Preserve important facts and qualifications, remove overlap and repetition, "
    "and do not add information absent from the source summaries. "
    "Use at most {max_words} words. Do not pad."
)

# Shared temperature for chunk, reduction, and final summary calls (0.0-1.0)
SUMMARY_TEMPERATURE = 0.3
