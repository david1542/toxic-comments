import os

from transformers.integrations import ClearMLCallback


class CustomClearMLCallback(ClearMLCallback):
    def on_save(self, *args, **kwargs):
        should_report_model = os.getenv("CLEARML_DISABLE_MODEL_UPLOADING", "TRUE")
        if should_report_model == "TRUE":
            super().on_save(*args, **kwargs)
