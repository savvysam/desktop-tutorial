import React, { useState, useRef } from "react";
import { Upload, Image as ImageIcon, ChevronDown, CheckCircle2, ArrowRight, Info, Palette } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

export function Hierarchy() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [seasonsOpen, setSeasonsOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile: File) => {
    setFile(selectedFile);
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
      setAnalysisComplete(false);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleAnalyze = () => {
    if (!file) return;
    setIsAnalyzing(true);
    // Simulate analysis delay
    setTimeout(() => {
      setIsAnalyzing(false);
      setAnalysisComplete(true);
    }, 2000);
  };

  const currentStep = analysisComplete ? 3 : file ? 2 : 1;

  const seasons = [
    { name: "Spring", colors: ["#F9A03F", "#F7D08A", "#E3F09B", "#87B6A7"], subtypes: ["Light", "Warm", "Clear"] },
    { name: "Summer", colors: ["#F7CAC9", "#F7CAC9", "#92A8D1", "#F7CAC9"], subtypes: ["Light", "Cool", "Soft"] },
    { name: "Autumn", colors: ["#C94C4C", "#92A8D1", "#88B04B", "#F7CAC9"], subtypes: ["Soft", "Warm", "Deep"] },
    { name: "Winter", colors: ["#34568B", "#FF6F61", "#6B5B95", "#88B04B"], subtypes: ["Clear", "Cool", "Deep"] },
  ];

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: `
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        .font-jakarta {
          font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .step-inactive { opacity: 0.4; pointer-events: none; transition: all 0.3s ease; }
        .step-active { opacity: 1; pointer-events: auto; transition: all 0.3s ease; }
        .step-completed { opacity: 0.7; transition: all 0.3s ease; }
      ` }} />
      <div className="min-h-screen bg-[#f8f7ff] font-jakarta text-[#1a1a2e] pb-24">
        
        {/* Header */}
        <header className="bg-white border-b border-[#e5e5f0] py-6 px-6 sticky top-0 z-10 shadow-sm">
          <div className="max-w-3xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-[#6C63FF] flex items-center justify-center text-white font-bold text-lg">C</div>
              <h1 className="text-xl font-bold tracking-tight">Chroma</h1>
            </div>
            <div className="text-sm font-medium text-[#6C63FF] bg-[#6C63FF]/10 px-3 py-1 rounded-full">
              Personal Color Analysis
            </div>
          </div>
        </header>

        <main className="max-w-2xl mx-auto px-6 pt-12 space-y-12">
          
          <div className="text-center mb-8">
            <h2 className="text-4xl font-extrabold tracking-tight mb-4">Discover Your True Colors</h2>
            <p className="text-lg text-gray-600 max-w-lg mx-auto">Upload a natural light photo of your face, and our AI will determine your perfect color palette in seconds.</p>
          </div>

          {/* Task Flow Container */}
          <div className="bg-white rounded-3xl p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-[#e5e5f0]/50 relative">
            
            {/* Step 1: Upload */}
            <div className={`relative pl-12 pb-12 border-l-2 border-[#e5e5f0] ${currentStep === 1 ? 'step-active' : 'step-completed border-[#6C63FF]'}`}>
              <div className={`absolute -left-[17px] top-0 w-8 h-8 rounded-full flex items-center justify-center font-bold border-2 bg-white
                ${currentStep > 1 ? 'border-[#6C63FF] text-[#6C63FF]' : currentStep === 1 ? 'border-[#6C63FF] bg-[#6C63FF] text-white' : 'border-gray-300 text-gray-400'}`}>
                {currentStep > 1 ? <CheckCircle2 className="w-5 h-5" /> : '1'}
              </div>
              
              <h3 className="text-2xl font-bold mb-6">Upload your photo</h3>
              
              <div 
                className={`relative border-2 border-dashed rounded-2xl p-10 transition-all text-center
                  ${isDragging ? 'border-[#6C63FF] bg-[#6C63FF]/5 scale-[1.02]' : 'border-[#e5e5f0] hover:border-gray-400 hover:bg-gray-50'}
                  ${file ? 'hidden' : 'block'}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  className="hidden" 
                  ref={fileInputRef} 
                  onChange={handleFileSelect} 
                  accept="image/jpeg,image/png,image/webp" 
                />
                <div className="mx-auto w-16 h-16 rounded-full bg-[#f8f7ff] flex items-center justify-center mb-4">
                  <Upload className="w-8 h-8 text-[#6C63FF]" />
                </div>
                <p className="text-lg font-semibold mb-2">Click to upload or drag and drop</p>
                <p className="text-sm text-gray-500 mb-6">SVG, PNG, JPG or GIF (max. 5MB)</p>
                <Button className="bg-[#6C63FF] hover:bg-[#5a52d5] text-white font-medium px-8 py-6 rounded-xl text-md">
                  Choose Photo
                </Button>
              </div>

              {preview && (
                <div className="flex items-center gap-6 p-4 border border-[#e5e5f0] rounded-2xl">
                  <div className="w-24 h-24 rounded-xl overflow-hidden bg-gray-100 flex-shrink-0 border border-gray-200">
                    <img src={preview} alt="Preview" className="w-full h-full object-cover" />
                  </div>
                  <div className="flex-1">
                    <p className="font-semibold text-lg">{file?.name}</p>
                    <p className="text-sm text-gray-500">{(file?.size ? (file.size / 1024 / 1024).toFixed(2) : 0)} MB</p>
                  </div>
                  <Button 
                    variant="outline" 
                    onClick={() => { setFile(null); setPreview(null); setAnalysisComplete(false); }}
                    className="border-gray-200 hover:bg-gray-50 text-gray-600 rounded-xl"
                  >
                    Change Photo
                  </Button>
                </div>
              )}
            </div>

            {/* Step 2: Analyze */}
            <div className={`relative pl-12 pb-12 border-l-2 border-[#e5e5f0] ${currentStep === 2 ? 'step-active' : currentStep > 2 ? 'step-completed border-[#6C63FF]' : 'step-inactive'}`}>
              <div className={`absolute -left-[17px] top-0 w-8 h-8 rounded-full flex items-center justify-center font-bold border-2 bg-white
                ${currentStep > 2 ? 'border-[#6C63FF] text-[#6C63FF]' : currentStep === 2 ? 'border-[#6C63FF] bg-[#6C63FF] text-white' : 'border-gray-300 text-gray-400'}`}>
                {currentStep > 2 ? <CheckCircle2 className="w-5 h-5" /> : '2'}
              </div>
              
              <h3 className="text-2xl font-bold mb-4">Run Analysis</h3>
              <p className="text-gray-600 mb-6">Our algorithm will analyze your skin undertone, eye color, and hair contrast.</p>
              
              <Button 
                onClick={handleAnalyze} 
                disabled={!file || isAnalyzing || analysisComplete}
                className="w-full bg-[#6C63FF] hover:bg-[#5a52d5] text-white font-bold text-lg py-8 rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-[#6C63FF]/20"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-6 h-6 border-4 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Analyzing facial features...
                  </>
                ) : analysisComplete ? (
                  <>
                    Analysis Complete <CheckCircle2 className="w-6 h-6" />
                  </>
                ) : (
                  <>
                    Reveal My Season <ArrowRight className="w-6 h-6" />
                  </>
                )}
              </Button>
            </div>

            {/* Step 3: Result */}
            <div className={`relative pl-12 ${currentStep === 3 ? 'step-active' : 'step-inactive border-transparent'}`}>
              <div className={`absolute -left-[17px] top-0 w-8 h-8 rounded-full flex items-center justify-center font-bold border-2 bg-white
                ${currentStep === 3 ? 'border-[#6C63FF] bg-[#6C63FF] text-white' : 'border-gray-300 text-gray-400'}`}>
                3
              </div>
              
              <h3 className="text-2xl font-bold mb-6">Your Result</h3>
              
              {analysisComplete ? (
                <div className="bg-[#f8f7ff] rounded-2xl p-8 text-center border border-[#6C63FF]/20 relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-[#6C63FF] to-[#a39eff]"></div>
                  <Palette className="w-12 h-12 text-[#6C63FF] mx-auto mb-4" />
                  <p className="text-sm font-bold tracking-widest text-[#6C63FF] uppercase mb-2">You are a</p>
                  <h4 className="text-4xl font-extrabold mb-4 text-[#1a1a2e]">True Winter</h4>
                  <p className="text-gray-600 mb-8 max-w-sm mx-auto">High contrast, cool undertones. You look stunning in vivid, icy colors and pure black and white.</p>
                  
                  <div className="flex justify-center gap-3 mb-6">
                    {['#E32636', '#2E4A62', '#1A1A1A', '#F0F0F0', '#B565A7', '#009B77'].map((color, i) => (
                      <div key={i} className="w-10 h-10 rounded-full shadow-inner border border-black/5" style={{ backgroundColor: color }}></div>
                    ))}
                  </div>
                  
                  <Button variant="outline" className="w-full border-[#6C63FF] text-[#6C63FF] hover:bg-[#6C63FF]/5 rounded-xl py-6 font-semibold">
                    View Full Palette & Guide
                  </Button>
                </div>
              ) : (
                <div className="h-32 rounded-2xl border-2 border-dashed border-gray-200 bg-gray-50 flex items-center justify-center text-gray-400">
                  <p className="font-medium">Complete steps 1 & 2 to see your result</p>
                </div>
              )}
            </div>
          </div>

          {/* Secondary Content: Collapsible Reference */}
          <div className="max-w-2xl mx-auto">
            <Collapsible open={seasonsOpen} onOpenChange={setSeasonsOpen} className="bg-white rounded-2xl shadow-sm border border-[#e5e5f0] overflow-hidden">
              <CollapsibleTrigger asChild>
                <button className="w-full flex items-center justify-between p-6 hover:bg-gray-50 transition-colors focus:outline-none">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-[#f8f7ff] flex items-center justify-center text-[#6C63FF]">
                      <Info className="w-5 h-5" />
                    </div>
                    <div className="text-left">
                      <h4 className="font-bold text-lg">The 12 Color Seasons</h4>
                      <p className="text-sm text-gray-500">Learn about the armocromia system</p>
                    </div>
                  </div>
                  <ChevronDown className={`w-6 h-6 text-gray-400 transition-transform duration-300 ${seasonsOpen ? 'rotate-180' : ''}`} />
                </button>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="px-6 pb-6 pt-2 border-t border-gray-100">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {seasons.map((season) => (
                      <div key={season.name} className="p-4 rounded-xl bg-gray-50 border border-gray-100">
                        <h5 className="font-bold mb-2">{season.name}</h5>
                        <p className="text-xs text-gray-500 mb-3">{season.subtypes.join(" • ")}</p>
                        <div className="flex gap-2">
                          {season.colors.map((color, i) => (
                            <div key={i} className="w-6 h-6 rounded-md shadow-sm" style={{ backgroundColor: color }}></div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>

        </main>
      </div>
    </>
  );
}
