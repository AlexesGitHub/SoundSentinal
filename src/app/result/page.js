'use client';

import Header from '../../../components/header';
import React, { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';

// 1. Data Definitions (The "Look")
const RESULTS_CONFIG = {
  unlikely:   { color: 'text-green-500',  bar: 'bg-green-500',  desc: 'This audio is Unlikely to have been AI generated.' },
  possibly:   { color: 'text-yellow-500', bar: 'bg-yellow-500', desc: 'We cannot be certain, but there is a chance this is AI generated.' },
  likely:     { color: 'text-orange-500', bar: 'bg-orange-500', desc: 'This audio is Likely to have been AI generated.' },
  veryLikely: { color: 'text-red-500',    bar: 'bg-red-500',    desc: 'This audio is Very Likely to have been AI generated.' }
};

// 2. The Logic Helper (The "Brain")
const getAnalysisCategory = (label, score) => {
  if (label !== "Deepfake Audio") return 'unlikely';
  if (score > 85) return 'veryLikely';
  if (score > 60) return 'likely';
  return 'possibly';
};

// 3. Helper to format the title (e.g., "veryLikely" -> "Very Likely")
const formatTitle = (str) => str.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());

function ResultContent() {
  const searchParams = useSearchParams();
  const label = searchParams.get('label');
  const score = parseFloat(searchParams.get('score')) || 0;

  const categoryKey = getAnalysisCategory(label, score);
  const config = RESULTS_CONFIG[categoryKey];

  return (
    <div className="w-full max-w-2xl text-center">
      {/* Main Result Heading */}
      <h1 className="text-4xl sm:text-5xl font-bold mb-4">
        Result: <span className={config.color}>{formatTitle(categoryKey)}</span>
      </h1>

      {/* Confidence Subtext */}
      <p className="text-xl font-semibold mb-6 opacity-80">
        AI Confidence Level: {score.toFixed(1)}%
      </p>

      {/* Description Box */}
      <p className="text-sm sm:text-lg text-gray-600 dark:text-gray-400 mb-10 max-w-lg mx-auto leading-relaxed">
        {config.desc}
      </p>

      {/* Visual Progress Bar */}
      <div className="w-full h-4 bg-gray-100 dark:bg-gray-800 rounded-full relative mb-12 shadow-inner overflow-hidden">
        <div 
          className={`absolute left-0 h-full ${config.bar} transition-all duration-1000 ease-out shadow-lg`} 
          style={{ width: `${score}%` }}
        />
      </div>

      {/* Action Button */}
      <Link href="/upload">
        <button className="bg-neutral-800 text-white font-medium py-4 px-10 rounded-full hover:bg-neutral-900 transition-all transform hover:scale-105 active:scale-95 shadow-md">
          Analyse a different clip
        </button>
      </Link>
    </div>
  );
}

// Final Export with Suspense (Required for useSearchParams in Next.js)
export default function ResultPage() {
  return (
    <div className="flex min-h-screen flex-col items-center bg-white dark:bg-black font-sans">
      <Header />
      <main className="flex flex-col items-center justify-start flex-1 w-full py-16 px-6">
        <Suspense fallback={<p>Loading results...</p>}>
          <ResultContent />
        </Suspense>
      </main>
    </div>
  );
}