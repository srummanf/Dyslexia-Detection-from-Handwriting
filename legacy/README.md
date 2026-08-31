# Legacy (2024) code

Preserved as-is from the original AI course project. **Not used by the current
app or notebook.** Kept for reference and provenance.

| Path | What it was |
|---|---|
| `app.py` | Original Streamlit app: Azure OCR (keys were committed — now invalidate them), 4 feature extractors, hand-transcribed decision tree, plus mic-based pronunciation / dictation / phonetics tabs. |
| `app2.py` | Abandoned YOLOv5 object-detection experiment. |
| `Detection_Of_Dyslexia_From_Handwriting.ipynb` | Colab notebook that trained `yolov8n-cls` on `data.zip`. |
| `model_training/` | Tabular model notebook (LogReg / DecisionTree / SVM) + `Decision_tree_model.sav` + train/test CSVs. |
| `model_results/` | Ultralytics run outputs (train, train2, train3). |
| `archive/` | Large raw artefacts removed from version control: `data.zip`, the `runs-*.zip` training dumps, `yolov8n-cls.pt`. |
| `temp.jpg` | Scratch file the old app wrote uploads to. |

## Security note

`app.py` contained live Azure Cognitive Services and Bing Spell-Check API keys.
If those subscriptions still exist, **rotate/revoke the keys** — they are in the
git history.
