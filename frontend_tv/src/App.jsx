import React, { useState, useEffect } from 'react';
import { Activity, CheckCircle2, AlertTriangle, AlertOctagon, Clock, Database, FileOutput, Server, Calendar, Target, TrendingUp, Layers, RotateCcw, Zap, Loader2 } from 'lucide-react';

export default function App() {
  const [data, setData] = useState(null);
  const [feedSpeed, setFeedSpeed] = useState(2); // 1: Thực tế, 2: Nhanh, 3: Siêu tốc
  const [dbConnected, setDbConnected] = useState(false); // Trạng thái kết nối database
  const [hasData, setHasData] = useState(false); // Trạng thái có dữ liệu feed

  // Fetch dữ liệu từ API (Tự động cập nhật nếu Backend đang bật)
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('/api/live-status');
        if (response.ok) {
          const result = await response.json();
          if (result.status === "success") {
            setData(result);
            setDbConnected(true);
            setHasData(true);
          } else {
            setHasData(false);
            setDbConnected(true);
          }
        } else {
          setDbConnected(false);
        }
      } catch (error) {
        setDbConnected(false);
      }
    };
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  // Show waiting screen if no data yet
  if (!hasData) {
    return (
      <div className="h-screen w-screen bg-[#070b14] text-slate-300 font-sans overflow-hidden flex items-center justify-center">
        <div className="text-center flex flex-col items-center gap-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-4 bg-transparent border-2 border-cyan-500/50 rounded-lg shadow-[0_0_15px_rgba(6,182,212,0.3)]">
              <Activity className="w-12 h-12 text-cyan-400" />
            </div>
            <div className="flex flex-col justify-center">
              <h1 className="text-3xl md:text-4xl font-black text-white tracking-wider drop-shadow-md">TRUNG TÂM GIÁM SÁT AI</h1>
              <p className="text-slate-400 text-xs md:text-sm mt-1">PdM 4.0 - Dự báo khả năng xảy ra sự cố</p>
            </div>
          </div>

          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-16 h-16 text-cyan-400 animate-spin" />
            <div className="text-xl md:text-2xl font-semibold text-cyan-300">ĐANG CHỜ DỮ LIỆU...</div>
            <div className="text-sm text-slate-500 max-w-md">
              Vui lòng chạy live_data_feeder để bắt đầu nhận dữ liệu từ các máy.
            </div>
            <div className="mt-8 text-xs text-slate-600 flex items-center gap-2">
              <Database className="w-4 h-4" />
              {dbConnected ? "Backend: Đã kết nối" : "Backend: Đang chờ kết nối..."}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-[#070b14] text-slate-300 font-sans overflow-hidden flex flex-col p-3 md:p-4 box-border">
      
      {/* 1. TOP HEADER */}
      <header className="flex justify-between items-center mb-4 shrink-0 relative">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-transparent border-2 border-cyan-500/50 rounded-lg shadow-[0_0_15px_rgba(6,182,212,0.3)]">
            <Activity className="w-8 h-8 text-cyan-400" />
          </div>
          <div className="flex flex-col justify-center">
            <h1 className="text-3xl md:text-3xl font-black text-white tracking-wider drop-shadow-md">TRUNG TÂM GIÁM SÁT AI (PdM 4.0)</h1>
            <p className="text-slate-400 text-xs md:text-sm mt-1">Dự báo khả năng xảy ra sự cố máy tự động trong 48h tiếp theo</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3 md:gap-4 text-sm md:text-base">
          {/* Database Connection Indicator */}
          <div className={`flex items-center gap-2 px-3 py-1.5 md:px-4 md:py-2 border rounded font-bold shadow-[inset_0_0_10px_rgba(34,197,94,0.1)] ${dbConnected ? 'border-green-500/50 text-green-400' : 'border-red-500/50 text-red-400'}`}>
            <Database className="w-4 h-4 md:w-5 md:h-5" />
            {dbConnected ? 'DB: Đã kết nối' : 'DB: Mất kết nối'}
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 md:px-4 md:py-2 border border-green-500/50 rounded text-green-400 font-bold shadow-[inset_0_0_10px_rgba(34,197,94,0.1)]" style={{ paddingLeft: "8px", paddingRight: "8px" }}>
            <CheckCircle2 className="w-4 h-4 md:w-5 md:h-5" /> ỔN ĐỊNH: {data.kpis.safe}
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 md:px-4 md:py-2 bg-[#1a1500] border border-yellow-500 rounded text-yellow-500 font-bold shadow-[inset_0_0_10px_rgba(234,179,8,0.2)]" style={{ paddingLeft: "8px", paddingRight: "8px" }}>
            <AlertTriangle className="w-4 h-4 md:w-5 md:h-5" /> CẢNH BÁO: {data.kpis.warn}
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 md:px-4 md:py-2 bg-[#2a0808] border border-red-500 rounded text-red-500 font-bold shadow-[inset_0_0_10px_rgba(239,68,68,0.3)]" style={{ paddingLeft: "8px", paddingRight: "8px" }}>
            <AlertOctagon className="w-4 h-4 md:w-5 md:h-5 animate-pulse" /> BÁO ĐỘNG: {data.kpis.danger}
          </div>
        </div>
      </header>

      {/*KHỐI DASHBOARD TRUNG TÂM */}
      <section className="grid grid-cols-12 gap-4 mb-4 shrink-0 h-[28vh]">
        
        {/* PANEL 1 SỰ CỐ THỰC TẾ */}
        <div className="col-span-3 bg-[#0d1522] rounded-xl border border-slate-700 p-3 md:p-4 flex flex-col relative shadow-lg">
          <h2 className="text-white font-bold flex items-center gap-2 mb-2 md:mb-4 text-sm md:text-base border-b border-slate-700/50 pb-2">
            <Calendar className="w-4 h-4 text-cyan-400"/> SỰ CỐ THỰC TẾ
          </h2>
          <div className="flex-1 flex items-center justify-between">
            {/* Donut Chart */}
            <div className="relative w-24 h-24 md:w-32 md:h-32 flex items-center justify-center">
              <svg viewBox="0 0 36 36" className="w-full h-full transform -rotate-90 drop-shadow-md">
                <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#1e293b" strokeWidth="4" />
                <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#22c55e" strokeWidth="4" strokeDasharray="56.3 43.7" />
                <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#eab308" strokeWidth="4" strokeDasharray="28.1 71.9" strokeDashoffset="-56.3" />
                <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#ef4444" strokeWidth="4" strokeDasharray="15.6 84.4" strokeDashoffset="-84.4" />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-2xl md:text-3xl font-black text-white">32</span>
                <span className="text-[9px] md:text-[11px] text-slate-400">Tổng sự cố</span>
              </div>
            </div>
            {/* Legend */}
            <div className="flex flex-col gap-2 md:gap-3 text-xs md:text-sm font-semibold w-[45%]">
              <div className="flex justify-between items-center"><span className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-green-500"></div> Đã xử lý</span><span className="text-white text-base">18</span></div>
              <div className="flex justify-between items-center"><span className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-yellow-500"></div> Đang xử lý</span><span className="text-white text-base">9</span></div>
              <div className="flex justify-between items-center"><span className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-red-500"></div> Chưa xử lý</span><span className="text-white text-base">5</span></div>
            </div>
          </div>
        </div>

        {/* PANEL 2 SỰ CỐ DỰ ĐOÁN */}
        <div className="col-span-3 bg-[#0d1522] rounded-xl border border-slate-700 p-3 md:p-4 flex flex-col relative shadow-lg">
          <h2 className="text-white font-bold flex items-center gap-2 mb-2 md:mb-4 text-sm md:text-base border-b border-slate-700/50 pb-2">
            <Target className="w-4 h-4 text-cyan-400"/> SỰ CỐ DỰ ĐOÁN
          </h2>
          <div className="flex-1 flex items-center justify-between">
            <div className="relative w-24 h-24 md:w-32 md:h-32 flex items-center justify-center">
              <svg viewBox="0 0 36 36" className="w-full h-full transform -rotate-90 drop-shadow-md">
                <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#1e293b" strokeWidth="4" />
                <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#22c55e" strokeWidth="4" strokeDasharray="75 25" />
                <circle cx="18" cy="18" r="15.915" fill="transparent" stroke="#ef4444" strokeWidth="4" strokeDasharray="25 75" strokeDashoffset="-75" />
              </svg>
              <div className="absolute flex flex-col items-center">
                <span className="text-2xl md:text-3xl font-black text-white">24</span>
                <span className="text-[9px] md:text-[11px] text-slate-400">Tổng dự đoán</span>
              </div>
            </div>
            <div className="flex flex-col gap-3 md:gap-4 text-xs md:text-sm font-semibold w-[45%]">
              <div className="flex justify-between items-center"><span className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-green-500"></div> Dự đoán đúng</span><span className="text-white text-base md:text-lg">18</span></div>
              <div className="flex justify-between items-center"><span className="flex items-center gap-2"><div className="w-2.5 h-2.5 rounded-full bg-red-500"></div> Dự đoán sai</span><span className="text-white text-base md:text-lg">6</span></div>
            </div>
          </div>
        </div>

        {/* PANEL 3 SO SÁNH VÀ ĐỘ TIN CẬY */}
        <div className="col-span-6 bg-[#0d1522] rounded-xl border border-slate-700 p-3 md:p-4 flex shadow-lg">
          {/* Cột Trái: Line Chart */}
          <div className="flex-1 flex flex-col border-r border-slate-700/50 pr-4">
            <div className="flex justify-between items-center mb-1 border-b border-slate-700/50 pb-2">
              <h2 className="text-white font-bold flex items-center gap-2 text-sm md:text-base"><TrendingUp className="w-4 h-4 text-cyan-400"/> SO SÁNH THỰC TẾ VÀ DỰ ĐOÁN</h2>
              <div className="flex gap-4 text-[10px] md:text-xs font-semibold">
                <span className="flex items-center gap-1.5 text-cyan-400"><div className="w-3 h-0.5 bg-cyan-400"></div> Dự đoán</span>
                <span className="flex items-center gap-1.5 text-yellow-500"><div className="w-3 h-0.5 bg-yellow-500"></div> Thực tế</span>
              </div>
            </div>
            <div className="flex-1 w-full relative pt-2 flex flex-col">
               <div className="text-[9px] text-slate-500 absolute top-2 left-0">Số sự cố</div>
               <svg viewBox="0 0 400 100" className="w-full h-full mt-2" preserveAspectRatio="none">
                  {/* Grid */}
                  <line x1="20" y1="10" x2="400" y2="10" stroke="#1e293b" strokeWidth="1" />
                  <line x1="20" y1="40" x2="400" y2="40" stroke="#1e293b" strokeWidth="1" />
                  <line x1="20" y1="70" x2="400" y2="70" stroke="#1e293b" strokeWidth="1" />
                  <line x1="20" y1="100" x2="400" y2="100" stroke="#1e293b" strokeWidth="1" />
                  {/* Y Axis labels */}
                  <text x="0" y="13" fill="#64748b" fontSize="10">40</text>
                  <text x="0" y="43" fill="#64748b" fontSize="10">30</text>
                  <text x="0" y="73" fill="#64748b" fontSize="10">20</text>
                  <text x="0" y="100" fill="#64748b" fontSize="10">10</text>
                  
                  {/* Lines */}
                  <path d="M 40,40 L 100,20 L 160,35 L 220,25 L 280,10 L 340,30 L 400,60" fill="none" stroke="#22d3ee" strokeWidth="1.5" />
                  <circle cx="40" cy="40" r="3" fill="#22d3ee" />
                  <circle cx="100" cy="20" r="3" fill="#22d3ee" />
                  <circle cx="160" cy="35" r="3" fill="#22d3ee" />
                  <circle cx="220" cy="25" r="3" fill="#22d3ee" />
                  <circle cx="280" cy="10" r="3" fill="#22d3ee" />
                  <circle cx="340" cy="30" r="3" fill="#22d3ee" />

                  <path d="M 40,65 L 100,45 L 160,60 L 220,50 L 280,30 L 340,60 L 400,80" fill="none" stroke="#eab308" strokeWidth="1.5" />
                  <circle cx="40" cy="65" r="3" fill="#eab308" />
                  <circle cx="100" cy="45" r="3" fill="#eab308" />
                  <circle cx="160" cy="60" r="3" fill="#eab308" />
                  <circle cx="220" cy="50" r="3" fill="#eab308" />
                  <circle cx="280" cy="30" r="3" fill="#eab308" />
                  <circle cx="340" cy="60" r="3" fill="#eab308" />
               </svg>
               <div className="flex justify-between text-[10px] text-slate-400 mt-1 pl-6">
                 <span>T1</span><span>T2</span><span>T3</span><span>T4</span><span>T5</span><span>T6</span><span>T7</span>
               </div>
            </div>
          </div>
          {/* Cột Phải: Metrics */}
          <div className="w-[30%] pl-4 flex flex-col justify-center gap-2">
            <h3 className="text-cyan-400 text-xs font-bold text-center mb-1 tracking-wide">ĐÁNH GIÁ ĐỘ TIN CẬY<br/>MÔ HÌNH</h3>
            <div className="flex justify-between items-center border border-cyan-900/50 p-2 rounded">
              <span className="text-xs text-slate-300">MAE</span><span className="text-white font-bold text-sm">2.4</span>
            </div>
            <div className="flex justify-between items-center border border-cyan-900/50 p-2 rounded">
              <span className="text-xs text-slate-300">MSE</span><span className="text-white font-bold text-sm">8.1</span>
            </div>
            <div className="flex justify-between items-center border border-cyan-900/50 p-2 rounded">
              <span className="text-xs text-slate-300">RMSE</span><span className="text-white font-bold text-sm">2.8</span>
            </div>
          </div>
        </div>
      </section>

      {/*LƯỚI THẺ MÁY MÓC */}
      <section className="flex-1 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3 overflow-hidden">
        {data.machines.map((m) => {
            const maxRisk = Math.max(...Object.values(m.alerts));
            const isDanger = maxRisk > 80;
            const isWarning = maxRisk > 40 && maxRisk <= 80;

            let borderStyle;
            let iconLines;
            let layersColor;
            let riskTextColor;
            let checkCircleColor;
            
            if (isDanger) {
                borderStyle = "border-red-600 bg-[#160606] shadow-[0_0_15px_rgba(220,38,38,0.3)]";
                iconLines = "text-red-500";
                layersColor = "text-red-400";
                riskTextColor = "text-red-400";
                checkCircleColor = "text-red-500/70";
            } else if (isWarning) {
                borderStyle = "border-yellow-500 bg-[#2a2000] shadow-[0_0_15px_rgba(234,179,8,0.3)]";
                iconLines = "text-yellow-500";
                layersColor = "text-yellow-400";
                riskTextColor = "text-yellow-400";
                checkCircleColor = "text-yellow-500/70";
            } else {
                borderStyle = "border-green-600/60 bg-[#091515] hover:border-green-400/80 transition-colors";
                iconLines = "text-green-500";
                layersColor = "text-blue-400";
                riskTextColor = "text-green-400";
                checkCircleColor = "text-green-500/70";
            }

            return (
              <div key={m.machine_id} className={`rounded-xl border p-2 flex flex-col relative ${borderStyle}`}>
                  
                {/* Header Card */}
                <div className="flex justify-between items-start mb-2 border-b border-slate-700/40 pb-2">
                  <div className="flex items-center gap-2">
                    <div className="w-18 h-16 bg-black/40 rounded flex items-center justify-center overflow-hidden shrink-0 border border-slate-700">
                        {m.image ? (
                          <img 
                            src={m.image} 
                            alt={`Máy ${m.machine_id}`} 
                            className="w-full h-full object-cover" 
                          />
                        ) : (
                          /*máy nào chưa có ảnh thì vẫn hiện icon như cũ */
                          <Layers className={`w-10 h-10 ${layersColor}`} />
                        )}
                    </div>
                    <div>
                      {/* <h3 className={`text-sm text-base md:text-lg font-black tracking-widest ${titleColor}`}>MÁY #{m.machine_id.toString().padStart(2, '0')}</h3> */}
                      <h3 className="text-sm md:text-[17px] font-black tracking-widest uppercase text-white">{m.machine_name || `MÁY ${m.machine_id.toString().padStart(2, '0')}`}</h3>
                      <p className="text-[11px] md:text-[13px] text-slate-400 mt-0.5">Vị trí: {m.location}</p>
                    </div>
                  </div>
                  <div className={`font-black text-lg md:text-xl tracking-tighter ${iconLines}`}>
                    {m.load_percentage !== undefined ? (
                      <div className="flex gap-1 items-end">
                        <div className={`w-1 h-4 md:w-1.5 md:h-5 rounded-sm transition-colors ${m.load_percentage >= 40 ? 'bg-yellow-400' : 'bg-slate-600'}`}></div>
                        <div className={`w-1 h-4 md:w-1.5 md:h-5 rounded-sm transition-colors ${m.load_percentage >= 40 ? 'bg-yellow-400' : 'bg-slate-600'}`}></div>
                        <div className={`w-1 h-4 md:w-1.5 md:h-5 rounded-sm transition-colors ${m.load_percentage >= 80 ? 'bg-green-400' : 'bg-slate-600'}`}></div>
                        <span className="text-[10px] md:text-xs ml-1 opacity-70">{m.load_percentage.toFixed(0)}%</span>
                      </div>
                    ) : (
                      '///'
                    )}
                  </div>
                </div>

                {/* Body Card */}
                {isDanger && m.explanations ? (
                  <div className="flex-1 flex flex-col mt-1">
                    <div className="text-center mb-2 animate-pulse">
                      <span className="text-red-500 font-bold text-2xl md:text-2xl flex items-center justify-center gap-1">
                        <AlertTriangle className="w-6 h-6" /> 
                        LỖI {m.worst_comp.toUpperCase()} ({maxRisk.toFixed(1)}%)
                      </span>
                    </div>
                    <div className="text-[10px] md:text-xs flex-1">
                      {m.explanations.map((reason, idx) => (
                        <div key={idx} className="flex items-start gap-1.5 mb-1.5">
                          <div className="w-1.5 h-1.5 rounded-full bg-red-500 mt-1 shrink-0 mt-1"></div>
                          <span className="text-slate-300 text-[13px] md:text-[14px]">{reason}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col justify-around gap-1">
                    {Object.entries(m.alerts).map(([comp, risk]) => (
                      <div key={comp} className="flex justify-between items-center text-[17px] md:text-[19px] font-mono">
                        <span className="flex items-center gap-1.5 text-slate-400">
                          <CheckCircle2 className={`w-5 h-5 ${checkCircleColor}`} />
                          {comp.toUpperCase()}
                        </span>
                        <span className={`font-bold text-sm md:text-lg ${riskTextColor}`}>{risk.toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
      </section>

      {/* 4. FOOTER */}
      <footer className="mt-3 pt-2 border-t border-slate-800 flex items-center text-[10px] md:text-xs text-slate-400 shrink-0">

  {/* LEFT */}
  <div className="flex gap-4 md:gap-6">
    <span className="flex items-center gap-1">
      <Database className="w-3.5 h-3.5 text-cyan-500" />
      Nguồn dữ liệu: IIoT Gateway
    </span>

    <span className="flex items-center gap-1">
      <Clock className="w-3.5 h-3.5 text-cyan-500" />
      Cập nhật: 1 phút trước
    </span>

    <span className="flex items-center gap-1">
      <Server className="w-3.5 h-3.5 text-cyan-500" />
      Tổng số máy:
      <strong className="text-white ml-1">{data.kpis.total}</strong>
    </span>
  </div>

  {/* RIGHT (TIME + BUTTON) */}
  <div className="flex items-center gap-4 ml-auto">
    {/* TIME */}
    <div className="flex items-center gap-2 text-cyan-400 font-mono">
      <Clock className="w-4 h-4 md:w-5 md:h-5" />
      {data.latest_time || new Date().toLocaleDateString('en-CA') + ' ' + new Date().toLocaleTimeString('en-GB', {
        hour: '2-digit',
        minute: '2-digit'
      })}
    </div>
    <button className="flex items-center gap-1.5 bg-transparent border border-slate-600 hover:border-cyan-400 px-3 py-1.5 rounded text-cyan-400 transition-colors">
      <FileOutput className="w-3.5 h-3.5" />
      Xuất báo cáo
    </button>
  </div>
</footer>
    </div>
  );
}
