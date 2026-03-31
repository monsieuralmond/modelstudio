# Model Studio

Teachable Machine inspired clone rebuilt with React, Vite, and TensorFlow.js.

## Run locally

```bash
cd /Users/almond/Documents/teachable-machine-clone
npm install --legacy-peer-deps
npm run dev
```

Then open the local Vite URL shown in the terminal.

## Included in this version

- React and Vite project structure for easier extension
- A more Teachable Machine-like hero, project selector, and editor layout
- Real TensorFlow.js image training using MobileNet embeddings
- Real speech-commands transfer learning for microphone classification
- Real MoveNet pose detection with a custom pose classifier
- Webcam, microphone, and pose sample collection with live predictions
- Editable class names, local project saving, and model export actions
- Refined responsive layout, floating motion, and export or FAQ sections

## Notes

- Image mode performs real in-browser transfer learning through TensorFlow.js.
- Audio mode uses `@tensorflow-models/speech-commands`, whose peer dependency range is older than the current TensorFlow.js line, so `npm install --legacy-peer-deps` is required.
- Pose mode uses `@tensorflow-models/pose-detection` with MoveNet and also includes the supporting MediaPipe package for bundling compatibility.
- `npm run build` and `npm run dev -- --host 127.0.0.1 --port 4173` were both verified in this environment after installing Node.js.
