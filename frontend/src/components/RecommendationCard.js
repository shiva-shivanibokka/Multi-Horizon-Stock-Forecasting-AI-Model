import React from 'react';
import { FaThumbsUp, FaThumbsDown, FaHandPaper, FaInfoCircle } from 'react-icons/fa';

const RecommendationCard = ({ recommendation }) => {
  // Handle case where recommendation is undefined or null
  if (!recommendation) {
    return (
      <div className="rounded-lg border p-6 bg-gray-50 border-gray-200">
        <div className="flex items-start space-x-4">
          <div className="flex-shrink-0">
            <FaInfoCircle className="h-8 w-8 text-gray-500" />
          </div>
          <div className="flex-1">
            <h3 className="text-xl font-bold text-gray-800">
              No Recommendation Available
            </h3>
            <p className="text-gray-600 mt-2">
              Unable to generate recommendation at this time.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const getRecommendationIcon = (rec) => {
    switch (rec) {
      case 'BUY':
        return <FaThumbsUp className="h-8 w-8 text-green-500" />;
      case 'SELL':
        return <FaThumbsDown className="h-8 w-8 text-red-500" />;
      case 'HOLD':
        return <FaHandPaper className="h-8 w-8 text-yellow-500" />;
      default:
        return <FaInfoCircle className="h-8 w-8 text-gray-500" />;
    }
  };

  const getRecommendationColor = (rec) => {
    switch (rec) {
      case 'BUY':
        return 'bg-green-50 border-green-200';
      case 'SELL':
        return 'bg-red-50 border-red-200';
      case 'HOLD':
        return 'bg-yellow-50 border-yellow-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const getRecommendationTextColor = (rec) => {
    switch (rec) {
      case 'BUY':
        return 'text-green-800';
      case 'SELL':
        return 'text-red-800';
      case 'HOLD':
        return 'text-yellow-800';
      default:
        return 'text-gray-800';
    }
  };

  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'High':
        return 'text-green-600';
      case 'Medium':
        return 'text-yellow-600';
      case 'Low':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  return (
    <div className={`rounded-lg border p-6 ${getRecommendationColor(recommendation.recommendation)}`}>
      <div className="flex items-start space-x-4">
        <div className="flex-shrink-0">
          {getRecommendationIcon(recommendation.recommendation)}
        </div>
        <div className="flex-1">
          <div className="flex items-center justify-between">
            <h3 className={`text-xl font-bold ${getRecommendationTextColor(recommendation.recommendation)}`}>
              {recommendation.recommendation}
            </h3>
            <span className={`text-sm font-medium ${getConfidenceColor(recommendation.confidence)}`}>
              Confidence: {recommendation.confidence}
            </span>
          </div>
          
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Analysis Score: {recommendation.score}</h4>
            <p className="text-xs text-gray-600 mb-3">
              <strong>What is the Analysis Score?</strong> This is a composite score that combines price predictions, technical indicators, and sentiment analysis. 
              Higher positive scores suggest stronger buy signals, while lower negative scores suggest stronger sell signals.
            </p>
            <div className="text-xs text-gray-600 mb-3">
              <strong>Score Interpretation:</strong>
              <ul className="mt-1 space-y-1">
                <li>• <strong>+3 and above:</strong> Strong BUY signal</li>
                <li>• <strong>+1 to +2:</strong> Moderate BUY signal</li>
                <li>• <strong>-1 to +1:</strong> HOLD recommendation</li>
                <li>• <strong>-1 to -2:</strong> Moderate SELL signal</li>
                <li>• <strong>-3 and below:</strong> Strong SELL signal</li>
              </ul>
            </div>
            
            {recommendation.reasons && recommendation.reasons.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Key Factors:</h4>
                <ul className="space-y-1">
                  {recommendation.reasons.map((reason, index) => (
                    <li key={index} className="text-sm text-gray-600 flex items-start">
                      <span className="inline-block w-2 h-2 bg-blue-400 rounded-full mt-2 mr-2 flex-shrink-0"></span>
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {recommendation.price_changes && Object.keys(recommendation.price_changes).length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Expected Price Changes:</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  {Object.entries(recommendation.price_changes).map(([timeframe, change]) => (
                    <div key={timeframe} className="flex justify-between">
                      <span className="text-gray-600">{timeframe}:</span>
                      <span className={change >= 0 ? 'text-green-600' : 'text-red-600'}>
                        {change >= 0 ? '+' : ''}{change.toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationCard; 