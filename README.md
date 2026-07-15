Terminal 1: python src/backend_api.py
Terminal 2: cd frontend_tv
npm run dev -- --host 
Terminal 3: python demo/data_pipeline/live_data_feeder.py

Trình duyệt laptop: http://localhost:5173
Android TV emulator: loadUrl("http://10.0.2.2:5173")

streamlit run src/frontend_tv.py
#node js
npm create vite@latest frontend_tv -- --template react
cd frontend_tv
npm install

npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p
npm install lucide-react

postgred cài timeScaleDB
bcp "i26s02004_iot_dev.dbo.data_ot" out "D:\export\data_ot.csv" -c -t, -S "10.29.134.73,45193" -U "i26s02004" -P "pfKJBmFdnQWrVqnJs"