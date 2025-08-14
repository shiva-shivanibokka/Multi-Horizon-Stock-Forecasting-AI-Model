import React, { useState } from 'react';
import { FaSmile, FaMeh, FaFrown, FaNewspaper, FaChevronDown, FaChevronUp } from 'react-icons/fa';

const SentimentAnalysis = ({ sentimentData }) => {
  const [showArticles, setShowArticles] = useState(false);

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return <FaSmile className="h-6 w-6 text-green-500" />;
      case 'negative':
        return <FaFrown className="h-6 w-6 text-red-500" />;
      case 'neutral':
        return <FaMeh className="h-6 w-6 text-yellow-500" />;
      default:
        return <FaMeh className="h-6 w-6 text-gray-500" />;
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'text-green-600';
      case 'negative':
        return 'text-red-600';
      case 'neutral':
        return 'text-yellow-600';
      default:
        return 'text-gray-600';
    }
  };

  const getSentimentBgColor = (sentiment) => {
    switch (sentiment) {
      case 'positive':
        return 'bg-green-50';
      case 'negative':
        return 'bg-red-50';
      case 'neutral':
        return 'bg-yellow-50';
      default:
        return 'bg-gray-50';
    }
  };

  const getScoreColor = (score) => {
    if (score > 0.1) return 'text-green-600';
    if (score < -0.1) return 'text-red-600';
    return 'text-yellow-600';
  };

  const getSentimentExplanation = (score) => {
    if (score > 0.5) return "Very Positive - Strong bullish sentiment with highly favorable news coverage";
    if (score > 0.2) return "Positive - Generally favorable news with optimistic outlook";
    if (score > 0.1) return "Slightly Positive - Mildly favorable sentiment";
    if (score > -0.1) return "Neutral - Balanced news coverage with mixed sentiment";
    if (score > -0.2) return "Slightly Negative - Mildly unfavorable sentiment";
    if (score > -0.5) return "Negative - Generally unfavorable news with concerning outlook";
    return "Very Negative - Strong bearish sentiment with highly unfavorable news coverage";
  };

  const getScoreExplanation = (score) => {
    return `Sentiment Score: ${score > 0 ? '+' : ''}${score} (Range: -1.0 to +1.0)`;
  };

  if (!sentimentData) {
    return (
      <div className="text-center py-8">
        <FaNewspaper className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-sm font-medium text-gray-900">
          No sentiment data available
        </h3>
        <p className="mt-1 text-sm text-gray-500">
          Unable to analyze market sentiment at this time.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Sentiment Summary */}
      <div className={`rounded-lg border p-6 ${getSentimentBgColor(sentimentData.sentiment)}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {getSentimentIcon(sentimentData.sentiment)}
            <div>
              <h3 className={`text-lg font-semibold ${getSentimentColor(sentimentData.sentiment)}`}>
                {sentimentData.sentiment.charAt(0).toUpperCase() + sentimentData.sentiment.slice(1)} Sentiment
              </h3>
              <p className="text-sm text-gray-600">
                Based on {sentimentData.articles_analyzed} recent news articles
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className={`text-2xl font-bold ${getScoreColor(sentimentData.score)}`}>
              {sentimentData.score > 0 ? '+' : ''}{sentimentData.score}
            </div>
            <div className="text-sm text-gray-500">Sentiment Score</div>
            <div className="text-xs text-gray-600 mt-1 max-w-xs">
              {getSentimentExplanation(sentimentData.score)}
            </div>
          </div>
        </div>
      </div>

      {/* Articles Section */}
      {sentimentData.articles && sentimentData.articles.length > 0 && (
        <div className="bg-white border rounded-lg">
          <button
            onClick={() => setShowArticles(!showArticles)}
            className="w-full px-6 py-4 text-left flex items-center justify-between hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <div className="flex items-center">
              <FaNewspaper className="h-5 w-5 text-blue-500 mr-2" />
              <span className="font-medium text-gray-900">
                Recent News Articles ({sentimentData.articles.length})
              </span>
            </div>
            {showArticles ? <FaChevronUp className="h-5 w-5 text-gray-400" /> : <FaChevronDown className="h-5 w-5 text-gray-400" />}
          </button>

          {showArticles && (
            <div className="border-t border-gray-200">
              <div className="max-h-96 overflow-y-auto">
                {sentimentData.articles.map((article, index) => (
                  <div key={index} className="px-6 py-4 border-b border-gray-100 last:border-b-0">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h4 className="text-sm font-medium text-gray-900 mb-1">
                          {article.title}
                        </h4>
                        <p className="text-sm text-gray-600 mb-2">
                          {article.summary}
                        </p>
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-gray-500">
                            {article.date}
                          </span>
                          <div className="flex items-center space-x-2">
                            <span className={`text-xs font-medium ${getScoreColor(article.sentiment_score)}`}>
                              {article.sentiment_score > 0 ? '+' : ''}{article.sentiment_score}
                            </span>
                            {article.link && (
                              <a 
                                href={article.link} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="text-xs text-blue-600 hover:text-blue-800 underline"
                              >
                                Read →
                              </a>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Sentiment Interpretation */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h4 className="text-sm font-medium text-blue-900 mb-2">
          Understanding Sentiment Analysis:
        </h4>
        <div className="text-sm text-blue-800 space-y-2">
          <p><strong>What is Sentiment Analysis?</strong> This analyzes the emotional tone of news articles about the company using Natural Language Processing (NLP).</p>
          <p><strong>Score Range:</strong> -1.0 (Very Negative) to +1.0 (Very Positive)</p>
          <ul className="space-y-1 mt-2">
            <li>• <strong>+0.5 to +1.0:</strong> Very Positive - Strong bullish sentiment</li>
            <li>• <strong>+0.2 to +0.5:</strong> Positive - Generally favorable news</li>
            <li>• <strong>+0.1 to +0.2:</strong> Slightly Positive - Mildly favorable</li>
            <li>• <strong>-0.1 to +0.1:</strong> Neutral - Balanced coverage</li>
            <li>• <strong>-0.2 to -0.1:</strong> Slightly Negative - Mildly unfavorable</li>
            <li>• <strong>-0.5 to -0.2:</strong> Negative - Generally unfavorable</li>
            <li>• <strong>-1.0 to -0.5:</strong> Very Negative - Strong bearish sentiment</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default SentimentAnalysis; 