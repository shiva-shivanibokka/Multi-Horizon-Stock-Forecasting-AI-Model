import React from 'react';
import { FaArrowUp, FaArrowDown, FaMinus } from 'react-icons/fa';

const PredictionTable = ({ predictions, currentPrice }) => {
  const timeframes = [
    { key: '1d', label: '1 Day', color: 'text-blue-600' },
    { key: '1w', label: '1 Week', color: 'text-green-600' },
    { key: '1m', label: '1 Month', color: 'text-purple-600' },
    { key: '6m', label: '6 Months', color: 'text-orange-600' },
    { key: '1y', label: '1 Year', color: 'text-red-600' }
  ];

  const getChangeIcon = (predictedPrice) => {
    const change = predictedPrice - currentPrice;
    if (change > 0) return <FaArrowUp className="text-green-500" />;
    if (change < 0) return <FaArrowDown className="text-red-500" />;
    return <FaMinus className="text-gray-500" />;
  };

  const getChangeColor = (predictedPrice) => {
    const change = predictedPrice - currentPrice;
    if (change > 0) return 'text-green-600';
    if (change < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  const getChangePercentage = (predictedPrice) => {
    return ((predictedPrice - currentPrice) / currentPrice * 100).toFixed(2);
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Timeframe
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Predicted Price
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Change
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              % Change
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {timeframes.map((timeframe) => {
            const predictedPrice = predictions[timeframe.key];
            if (!predictedPrice || typeof predictedPrice !== 'number') return null;

            return (
              <tr key={timeframe.key} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    <div className={`text-sm font-medium ${timeframe.color}`}>
                      {timeframe.label}
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-semibold text-gray-900">
                    ${predictedPrice.toFixed(2)}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center">
                    {getChangeIcon(predictedPrice)}
                    <span className={`ml-2 text-sm font-medium ${getChangeColor(predictedPrice)}`}>
                      ${Math.abs(predictedPrice - currentPrice).toFixed(2)}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`text-sm font-medium ${getChangeColor(predictedPrice)}`}>
                    {getChangePercentage(predictedPrice)}%
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default PredictionTable; 