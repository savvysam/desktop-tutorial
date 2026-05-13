import React, { useState, useRef, KeyboardEvent } from "react";
import { Upload, Image as ImageIcon, AlertCircle, X, ArrowRight, Info } from "lucide-react";

export function Accessibility() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const validateFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith("image/")) {
      setError("Please upload a valid image file (JPEG, PNG).");
      return false;
    }
    if (selectedFile.size > 5 * 1024 * 1024) {
      setError("File size must be less than 5MB.");
      return false;
    }
    setError(null);
    return true;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && validateFile(droppedFile)) {
      setFile(droppedFile);
      setPreviewUrl(URL.createObjectURL(droppedFile));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && validateFile(selectedFile)) {
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
    }
  };

  const clearFile = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  // High contrast color mappings for seasons
  const seasons = [
    { name: "Light Spring", desc: "Warm & Light", hex: "#FFF1C1", textClass: "text-[#4A3B00]" },
    { name: "Warm Spring", desc: "Warm & Bright", hex: "#FCA311", textClass: "text-[#4A2600]" },
    { name: "Bright Spring", desc: "Bright & Warm", hex: "#F15BB5", textClass: "text-[#4A002C]" },
    { name: "Light Summer", desc: "Cool & Light", hex: "#D8E2DC", textClass: "text-[#002B36]" },
    { name: "Cool Summer", desc: "Cool & Muted", hex: "#8ECAE6", textClass: "text-[#002633]" },
    { name: "Soft Summer", desc: "Muted & Cool", hex: "#B5A6A5", textClass: "text-[#2B1B1A]" },
    { name: "Soft Autumn", desc: "Muted & Warm", hex: "#D4A373", textClass: "text-[#4A2800]" },
    { name: "Warm Autumn", desc: "Warm & Muted", hex: "#CA6702", textClass: "text-[#3D1E00]" },
    { name: "Deep Autumn", desc: "Deep & Warm", hex: "#AE2012", textClass: "text-[#3D0A00]" },
    { name: "Deep Winter", desc: "Deep & Cool", hex: "#003049", textClass: "text-[#FFFFFF]" },
    { name: "Cool Winter", desc: "Cool & Bright", hex: "#14213D", textClass: "text-[#FFFFFF]" },
    { name: "Bright Winter", desc: "Bright & Cool", hex: "#023E8A", textClass: "text-[#FFFFFF]" },
  ];

  return (
    <div className="min-h-screen bg-[#F4F4F5] text-[#111111] font-sans selection:bg-[#0056D2] selection:text-white pb-24">
      {/* Import an accessible font if possible, sticking to strong system fonts for safety */}
      <style>{`
        :root {
          --focus-ring: 0 0 0 4px #FFFFFF, 0 0 0 8px #0056D2;
        }
        .focus-visible-ring:focus-visible {
          outline: none;
          box-shadow: var(--focus-ring);
        }
      `}</style>

      <main className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 pt-12">
        <header className="mb-8">
          <h1 className="text-4xl font-extrabold tracking-tight mb-4 text-[#000000]">
            Discover Your Color Season
          </h1>
          <p className="text-lg leading-relaxed text-[#27272A] max-w-2xl">
            Upload a clear photo of your face in natural light. We will analyze your skin undertone, eye color, and hair color to determine which of the 12 color seasons suits you best.
          </p>
        </header>

        <section aria-labelledby="upload-heading" className="bg-white rounded-xl shadow-sm border-2 border-[#D4D4D8] p-6 sm:p-8 mb-12">
          <h2 id="upload-heading" className="text-2xl font-bold mb-6 text-[#000000]">
            Step 1: Upload Your Photo
          </h2>

          {!file ? (
            <div className="space-y-4">
              <div
                className={`relative border-4 border-dashed rounded-lg p-8 sm:p-12 text-center transition-colors
                  ${isDragging ? "border-[#0056D2] bg-[#EFF6FF]" : "border-[#71717A] bg-[#FAFAFA] hover:bg-[#F4F4F5]"}
                  focus-visible-ring`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={handleKeyDown}
                role="button"
                tabIndex={0}
                aria-label="Drag and drop your photo here, or press Enter to choose a file"
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept="image/jpeg, image/png"
                  className="sr-only"
                  id="photo-upload"
                  aria-describedby="upload-instructions file-error"
                  aria-invalid={!!error}
                />
                <div className="flex flex-col items-center justify-center space-y-4 pointer-events-none">
                  <div className="bg-white p-4 rounded-full shadow-sm border-2 border-[#E4E4E7]" aria-hidden="true">
                    <Upload className="w-8 h-8 text-[#0056D2]" strokeWidth={2.5} />
                  </div>
                  <div>
                    <p className="text-xl font-bold text-[#111111]" id="upload-instructions">
                      Drag and drop your photo here
                    </p>
                    <p className="text-base text-[#3F3F46] mt-2">
                      or
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent double trigger
                      fileInputRef.current?.click();
                    }}
                    className="mt-4 px-6 py-3 bg-white border-2 border-[#0056D2] text-[#0056D2] font-bold text-lg rounded-md hover:bg-[#EFF6FF] focus-visible-ring min-h-[44px] min-w-[120px] transition-colors"
                  >
                    Choose File
                  </button>
                  <p className="text-sm text-[#52525B] mt-4 font-medium">
                    Accepted formats: JPEG, PNG. Maximum size: 5MB.
                  </p>
                </div>
              </div>

              {error && (
                <div 
                  id="file-error" 
                  role="alert" 
                  className="flex items-start gap-3 p-4 bg-[#FEF2F2] border-l-4 border-[#DC2626] rounded-r-md text-[#991B1B]"
                >
                  <AlertCircle className="w-6 h-6 flex-shrink-0" aria-hidden="true" />
                  <p className="text-base font-medium">{error}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-6 p-6 border-2 border-[#D4D4D8] rounded-lg bg-[#FAFAFA]">
                <div className="w-24 h-24 flex-shrink-0 rounded-md overflow-hidden border-2 border-[#D4D4D8] bg-white">
                  {previewUrl ? (
                    <img src={previewUrl} alt="Preview of your uploaded photo" className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-[#F4F4F5]">
                      <ImageIcon className="w-8 h-8 text-[#71717A]" aria-hidden="true" />
                    </div>
                  )}
                </div>
                
                <div className="flex-grow">
                  <h3 className="text-lg font-bold text-[#111111] break-all">
                    {file.name}
                  </h3>
                  <p className="text-base text-[#52525B] mt-1">
                    {(file.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>

                <button
                  type="button"
                  onClick={clearFile}
                  className="w-full sm:w-auto px-5 py-3 border-2 border-[#DC2626] text-[#DC2626] font-bold text-base rounded-md hover:bg-[#FEF2F2] focus-visible-ring min-h-[44px] flex items-center justify-center gap-2 transition-colors"
                  aria-label={`Remove file ${file.name}`}
                >
                  <X className="w-5 h-5" aria-hidden="true" strokeWidth={2.5} />
                  Remove
                </button>
              </div>

              <div className="pt-4 border-t-2 border-[#E4E4E7]">
                <h2 className="text-xl font-bold mb-4 text-[#000000]">
                  Step 2: Proceed to Analysis
                </h2>
                <button
                  type="button"
                  className="w-full sm:w-auto px-8 py-4 bg-[#0056D2] text-white font-bold text-lg rounded-md hover:bg-[#0047AB] focus-visible-ring min-h-[44px] flex items-center justify-center gap-3 transition-colors shadow-sm"
                >
                  Analyze Photo
                  <ArrowRight className="w-6 h-6" aria-hidden="true" strokeWidth={2.5} />
                </button>
              </div>
            </div>
          )}
        </section>

        <section aria-labelledby="reference-heading">
          <div className="flex items-center gap-3 mb-6">
            <h2 id="reference-heading" className="text-2xl font-bold text-[#000000]">
              The 12 Color Seasons
            </h2>
            <div className="relative group flex items-center">
              <button 
                type="button" 
                className="p-2 text-[#0056D2] hover:bg-[#EFF6FF] rounded-full focus-visible-ring"
                aria-label="More information about color seasons"
              >
                <Info className="w-6 h-6" aria-hidden="true" />
              </button>
            </div>
          </div>
          
          <p className="text-lg text-[#27272A] mb-8">
            Seasonal color analysis categorizes human coloring into four main seasons, each divided into three subtypes based on your dominant characteristic.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6" role="list">
            {seasons.map((season) => (
              <div 
                key={season.name} 
                className="border-2 border-[#D4D4D8] rounded-lg p-5 bg-white flex flex-col h-full shadow-sm hover:border-[#0056D2] transition-colors focus-within:border-[#0056D2] focus-within:ring-4 focus-within:ring-[#0056D2] focus-within:ring-opacity-50"
                role="listitem"
                tabIndex={0}
              >
                <div className="flex items-center gap-4 mb-3">
                  <div 
                    className="w-12 h-12 rounded-full border-2 border-[#A1A1AA] flex-shrink-0 shadow-inner"
                    style={{ backgroundColor: season.hex }}
                    aria-hidden="true"
                  />
                  <h3 className="text-lg font-bold text-[#111111]">
                    {season.name}
                  </h3>
                </div>
                <div className={`mt-auto inline-block px-3 py-2 rounded-md font-semibold text-base border border-current ${season.textClass}`} style={{ backgroundColor: season.hex }}>
                  {season.desc}
                </div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
