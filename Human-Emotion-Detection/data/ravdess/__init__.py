# This init file marks the 'ravdess' directory as a Python package.
# You can add helper functions for loading or accessing RAVDESS files here if needed.

def get_ravdess_path():
    """
    Returns the relative path to the RAVDESS dataset directory.
    """
    import os
    return os.path.join(os.path.dirname(__file__), 'audio_speech_actors_01-24')
