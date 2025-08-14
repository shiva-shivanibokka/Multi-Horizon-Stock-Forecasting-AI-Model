import React from 'react';
import { FaSpinner } from 'react-icons/fa';

const LoadingSpinner = () => {
  return (
    <div className="flex justify-center items-center py-12">
      <div className="text-center">
        <FaSpinner className="mx-auto h-12 w-12 text-primary-600 animate-spin" />
        <h3 className="mt-4 text-lg font-medium text-gray-900">
          Analyzing stock data...
        </h3>
        <p className="mt-2 text-sm text-gray-500">
          This may take a few moments
        </p>
      </div>
    </div>
  );
};

export default LoadingSpinner; 