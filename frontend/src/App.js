import React, { useState } from 'react';
import axios from 'axios';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { 
  FaSearch, 
  FaChartLine, 
  FaLightbulb, 
  FaArrowUp, 
  FaArrowDown, 
  FaMinus,
  FaSpinner,
  FaBuilding
} from 'react-icons/fa';
import PredictionTable from './components/PredictionTable';
import RecommendationCard from './components/RecommendationCard';

import CompanyFundamentals from './components/CompanyFundamentals';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  const [ticker, setTicker] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSearch = async () => {
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol');
      return;
    }

    setLoading(true);
    setError('');
    setData(null);

    try {
      const response = await axios.get(`/api/predict/${ticker.trim()}`);
      
      // Check if the response has an error
      if (response.data.error) {
        setError(response.data.error);
        return;
      }
      
      // Validate the response data structure
      if (!response.data.predictions || !response.data.current_price) {
        setError('Invalid data received from server');
        return;
      }
      
      // Ensure all required data is present with deep validation
      const validatedData = {
        ticker: response.data.ticker || ticker.toUpperCase(),
        current_price: response.data.current_price || 0,
        predictions: response.data.predictions || {},
        fundamentals: response.data.fundamentals || {},
        technical_data: response.data.technical_data || {
          close: [],
          sma_50: [],
          sma_200: [],
          dates: []
        },
        recommendation: response.data.recommendation || {
          recommendation: "HOLD",
          confidence: "Low",
          score: 0,
          reasons: [],
          price_changes: {}
        }
      };
      
      // Add a small delay to prevent rendering issues
      setTimeout(() => {
        setData(validatedData);
      }, 50);
      
    } catch (err) {
      console.error('API Error:', err);
      setError(err.response?.data?.error || 'Failed to fetch data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  const formatChartData = (technicalData) => {
    if (!technicalData || !technicalData.dates) return [];
    
    try {
      const chartData = [];
      const maxLength = Math.min(
        technicalData.dates.length,
        technicalData.close.length,
        technicalData.sma_50.length,
        technicalData.sma_200.length
      );
      
      for (let i = 0; i < maxLength; i++) {
        chartData.push({
          date: new Date(technicalData.dates[i]).toLocaleDateString(),
          'Close Price': Number(technicalData.close[i]) || 0,
          'SMA 50': Number(technicalData.sma_50[i]) || 0,
          'SMA 200': Number(technicalData.sma_200[i]) || 0
        });
      }
      
      return chartData;
    } catch (error) {
      console.error('Error formatting chart data:', error);
      return [];
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <FaChartLine className="h-8 w-8 text-primary-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">
                Stock Prediction Dashboard
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <input
                  type="text"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  onKeyPress={handleKeyPress}
                  placeholder="Enter ticker symbol (e.g., AAPL)"
                  className="w-64 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-primary-600 disabled:opacity-50"
                >
                  {loading ? <FaSpinner className="animate-spin" /> : <FaSearch />}
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                 {error && (
           <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
             <div className="flex">
               <div className="flex-shrink-0">
                 <FaArrowDown className="h-5 w-5 text-red-400" />
               </div>
               <div className="ml-3">
                 <h3 className="text-sm font-medium text-red-800">
                   Error
                 </h3>
                 <div className="mt-2 text-sm text-red-700">
                   {error}
                 </div>
               </div>
             </div>
           </div>
         )}

                 {loading && <LoadingSpinner />}

         {data && !loading && (
           <ErrorBoundary>
             <div className="space-y-8">
               {/* Current Price and Ticker Info */}
               <div className="bg-white rounded-lg shadow p-6">
                 <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold text-gray-900">
                    {data.ticker}
                  </h2>
                  <p className="text-lg text-gray-600">
                    Current Price: ${data.current_price}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">Last Updated</p>
                  <p className="text-sm font-medium text-gray-900">
                    {new Date().toLocaleDateString()}
                  </p>
                </div>
              </div>
            </div>

            {/* Predictions Table */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">
                  Price Predictions
                </h3>
              </div>
              <div className="p-6">
                <PredictionTable 
                  predictions={data.predictions} 
                  currentPrice={data.current_price}
                />
              </div>
            </div>

            {/* Recommendation */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <div className="flex items-center">
                  <FaLightbulb className="h-5 w-5 text-yellow-500 mr-2" />
                  <h3 className="text-lg font-medium text-gray-900">
                    AI Recommendation
                  </h3>
                </div>
              </div>
              <div className="p-6">
                <RecommendationCard recommendation={data.recommendation} />
              </div>
            </div>

            {/* Chart */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h3 className="text-lg font-medium text-gray-900">
                  Price Chart with Moving Averages
                </h3>
              </div>
              <div className="p-6">
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={formatChartData(data.technical_data)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="date" 
                        tick={{ fontSize: 12 }}
                        angle={-45}
                        textAnchor="end"
                        height={80}
                      />
                      <YAxis tick={{ fontSize: 12 }} />
                                             <Tooltip 
                         formatter={(value, name) => {
                           const formattedValue = `$${value.toFixed(2)}`;
                           return [formattedValue, name];
                         }}
                         labelFormatter={(label) => `Date: ${label}`}
                         contentStyle={{
                           backgroundColor: 'rgba(255, 255, 255, 0.95)',
                           border: '1px solid #ccc',
                           borderRadius: '8px',
                           padding: '8px'
                         }}
                         cursor={{ stroke: '#3b82f6', strokeWidth: 2 }}
                       />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="Close Price" 
                        stroke="#3b82f6" 
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="SMA 50" 
                        stroke="#22c55e" 
                        strokeWidth={1.5}
                        dot={false}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="SMA 200" 
                        stroke="#f59e0b" 
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>



            {/* Company Fundamentals */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <div className="flex items-center">
                  <FaBuilding className="h-5 w-5 text-purple-500 mr-2" />
                  <h3 className="text-lg font-medium text-gray-900">
                    Company Fundamentals
                  </h3>
                </div>
              </div>
              <div className="p-6">
                <CompanyFundamentals fundamentals={data.fundamentals} />
              </div>
            </div>
          </div>
        </ErrorBoundary>
        )}

        {!data && !loading && !error && (
          <div className="text-center py-12">
            <FaChartLine className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">
              No data to display
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              Enter a ticker symbol above to get started.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 