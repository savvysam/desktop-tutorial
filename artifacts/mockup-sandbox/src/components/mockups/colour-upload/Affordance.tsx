import React, { useState, useRef } from 'react';
import { UploadCloud, ArrowDown, Image as ImageIcon, Sparkles, X, CheckCircle2, Search } from 'lucide-react';
import './_group.css';

export function Affordance() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (f: File) => {
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
      alert("Analysis complete! (Mock)");
    }, 2000);
  };

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const triggerFileSelect = () => {
    if (!file) {
      inputRef.current?.click();
    }
  };

  const seasons = [
    { season: "Spring", subtypes: ["Light", "Warm", "Clear"], color: "bg-green-100 text-green-900 border-green-300" },
    { season: "Summer", subtypes: ["Light", "Cool", "Soft"], color: "bg-blue-100 text-blue-900 border-blue-300" },
    { season: "Autumn", subtypes: ["Soft", "Warm", "Deep"], color: "bg-orange-100 text-orange-900 border-orange-300" },
    { season: "Winter", subtypes: ["Clear", "Cool", "Deep"], color: "bg-purple-100 text-purple-900 border-purple-300" }
  ];

  return (
    <div className="affordance-theme min-h-screen py-12 px-4 sm:px-6 lg:px-8" style={{ backgroundColor: 'var(--bg-color)', color: 'var(--text-main)' }}>
      <div className="max-w-3xl mx-auto space-y-12">
        
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-indigo-100 rounded-full mb-2">
            <Sparkles className="w-8 h-8 text-indigo-600" />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight">Discover Your True Colors</h1>
          <p className="text-xl max-w-2xl mx-auto" style={{ color: 'var(--text-muted)' }}>
            Upload a clear, well-lit photo of your face. We'll analyze your features to find your perfect seasonal color palette.
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white rounded-[32px] p-6 md:p-10 shadow-xl border border-gray-100">
          
          <input 
            type="file" 
            ref={inputRef}
            onChange={handleChange}
            accept="image/*"
            className="hidden" 
          />

          <div 
            className={`drop-zone relative border-4 border-dashed rounded-3xl p-8 md:p-16 flex flex-col items-center justify-center text-center cursor-pointer min-h-[350px]
              ${dragActive ? "active" : "border-gray-300"} 
              ${!file && !dragActive ? "animate-pulse-border" : ""}
              ${file ? "has-file border-transparent bg-gray-50" : ""}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={triggerFileSelect}
          >
            {!file ? (
              <div className="space-y-8 flex flex-col items-center pointer-events-none">
                <div className="bg-indigo-50 p-6 rounded-full animate-bounce-arrow">
                  <ArrowDown className="w-12 h-12 text-indigo-600" />
                </div>
                
                <div className="space-y-2">
                  <h3 className="text-2xl font-bold text-gray-900">Drag & Drop your photo here</h3>
                  <p className="text-lg text-gray-500 font-medium">JPEG, PNG or HEIC up to 10MB</p>
                </div>

                <div className="flex items-center w-full max-w-xs my-4">
                  <div className="flex-1 h-px bg-gray-300"></div>
                  <span className="px-4 text-gray-500 font-bold uppercase tracking-wider text-sm">OR</span>
                  <div className="flex-1 h-px bg-gray-300"></div>
                </div>

                <button 
                  type="button"
                  className="btn-secondary pointer-events-auto flex items-center justify-center gap-3 w-full max-w-xs py-4 px-8 rounded-2xl text-lg font-bold"
                  onClick={(e) => {
                    e.stopPropagation();
                    triggerFileSelect();
                  }}
                >
                  <Search className="w-5 h-5" />
                  Browse Files
                </button>
              </div>
            ) : (
              <div className="w-full h-full flex flex-col items-center pointer-events-auto">
                <div className="relative w-full max-w-md aspect-[3/4] md:aspect-square mb-8 rounded-2xl overflow-hidden shadow-lg ring-4 ring-indigo-50">
                  <img 
                    src={previewUrl!} 
                    alt="Preview" 
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                  <button 
                    onClick={removeFile}
                    className="absolute top-4 right-4 bg-white/90 hover:bg-white text-gray-900 p-2 rounded-full shadow-sm transition-all hover:scale-110"
                    aria-label="Remove image"
                  >
                    <X className="w-5 h-5" />
                  </button>
                  <div className="absolute bottom-4 left-4 flex items-center gap-2 text-white bg-black/30 backdrop-blur-md py-1.5 px-3 rounded-full text-sm font-medium">
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                    Photo selected
                  </div>
                </div>

                <button 
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="btn-primary w-full max-w-md py-5 px-8 rounded-2xl text-xl font-bold flex items-center justify-center gap-3 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-6 w-6 border-4 border-white border-t-transparent"></div>
                      Analyzing Features...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-6 h-6" />
                      Analyze My Colors Now
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Reference List */}
        <div className="pt-8 border-t border-gray-200">
          <h2 className="text-2xl font-bold text-center mb-8">The 12 Color Seasons</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {seasons.map((s) => (
              <div key={s.season} className={`p-6 rounded-2xl border-2 ${s.color} bg-opacity-50 backdrop-blur-sm`}>
                <h3 className="text-xl font-extrabold mb-4 flex items-center gap-2">
                  {s.season}
                </h3>
                <div className="flex flex-wrap gap-2">
                  {s.subtypes.map(sub => (
                    <span key={sub} className="px-3 py-1.5 bg-white/60 rounded-lg text-sm font-bold shadow-sm">
                      {sub} {s.season}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
