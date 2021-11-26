from datasets import load_dataset, GenerateMode

#dataset = load_dataset("AI-Sweden/SuperLim",  use_auth_token="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", download_mode=GenerateMode.FORCE_REDOWNLOAD)
dataset = load_dataset("AI-Sweden/SuperLim", name="SweWsc", data_files="SweWsc/test.jsonl", use_auth_token="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")
#dataset = load_dataset("wzkariampuzha/EpiClassifySet")
print("done!")
