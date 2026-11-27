# Short demo recording outline (about 90 seconds)

This is a narration and screen-capture checklist for recording the submitted demo video. The assessment deliverable is a video file; this outline is not itself a video.

Before recording, start Ollama, confirm `gemma4:12b` is available, start Fieldnotes, and wait until its page is open. In Windows Snipping Tool, choose **Record → New**, select the browser area, and record the steps below. Stop and save the recording as `demo.mp4` in this project folder.

1. **0:00–0:10:** Show the home page and explain that the goal is a source-grounded research prototype.
2. **0:10–0:25:** Enter “What is the latest stable version of Python? Link to the official page.” Point out the subquestions, official-source search, and live progress.
3. **0:25–0:45:** Show the short answer and important-claim cards. Point out that each claim links to its source, and open one link.
4. **0:45–1:05:** Search “How many people live in India? Show the dates if sources give different numbers.” Show the date context and possible-discrepancy note.
5. **1:05–1:20:** Search a made-up product, e.g. “What are the published clinical-trial results for the Zorblax home fusion battery?” Show the honest non-answer when relevant evidence is not found.
6. **1:20–1:30:** State the main limitation: the model can misread evidence, and the user must check source context. Briefly show the architecture and passing tests (`.venv\Scripts\python.exe -m unittest -v`).

For a stable recording, run locally with internet access and the model ready, wait for each result to finish, and keep the browser window free of personal tabs or notifications. Search-provider or model setup errors should be fixed before recording rather than cut from the demonstration.
