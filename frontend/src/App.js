import React, { useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell,
} from 'recharts';
import {
  FaSearch, FaChartLine, FaLightbulb,
  FaSpinner, FaBuilding, FaNewspaper,
} from 'react-icons/fa';
import PredictionTable    from './components/PredictionTable';
import RecommendationCard from './components/RecommendationCard';
import CompanyFundamentals from './components/CompanyFundamentals';
import SentimentAnalysis   from './components/SentimentAnalysis';
import LoadingSpinner      from './components/LoadingSpinner';
import ErrorBoundary       from './components/ErrorBoundary';

const MODELS = ['transformer', 'lstm', 'rnn', 'rf'];
const MODEL_COLORS = {
  transformer: '#3b82f6',
  lstm:        '#22c55e',
  rnn:         '#f59e0b',
  rf:          '#a855f7',
};
const HORIZON_LABELS = { '1d': '1 Day', '1w': '1 Week', '1m': '1 Month', '6m': '6 Months', '1y': '1 Year' };

function App() {
  const [ticker,    setTicker]    = useState('');
  const [model,     setModel]     = useState('transformer');
  const [activeTab, setActiveTab] = useState('forecast');
  const [data,      setData]      = useState(null);
  const [allData,   setAllData]   = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState('');

  const handleSearch = async () => {
    if (!ticker.trim()) { setError('Please enter a ticker symbol.'); return; }
    setLoading(true);
    setError('');
    setData(null);
    setAllData(null);
    setSentiment(null);

    try {
      // Run single-model, comparison, sentiment, and fundamentals in parallel
      const [single, all, sent] = await Promise.all([
        axios.get(`/api/predict/${ticker.trim()}?model=${model}`),
        axios.get(`/api/predict/all/${ticker.trim()}`),
        axios.get(`/api/sentiment/${ticker.trim()}`),
      ]);

      if (single.data.error) { setError(single.data.error); return; }
      setData(single.data);
      setAllData(all.data);
      setSentiment(sent.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to fetch data. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => { if (e.key === 'Enter') handleSearch(); };

  // Formats price history into the shape recharts expects
  const formatChartData = (technicalData) => {
    if (!technicalData?.dates) return [];
    const len = Math.min(
      technicalData.dates.length,
      technicalData.close.length,
      technicalData.sma_50.length,
      technicalData.sma_200.length,
    );
    return Array.from({ length: len }, (_, i) => ({
      date:          new Date(technicalData.dates[i]).toLocaleDateString(),
      'Close Price': Number(technicalData.close[i])   || 0,
      'SMA 50':      Number(technicalData.sma_50[i])  || 0,
      'SMA 200':     Number(technicalData.sma_200[i]) || 0,
    }));
  };

  // Turns the /predict/all response into rows for the comparison bar chart
  const buildComparisonData = (allData) => {
    if (!allData?.predictions) return [];
    return Object.keys(HORIZON_LABELS).map((h) => {
      const row = { horizon: HORIZON_LABELS[h] };
      MODELS.forEach((m) => {
        if (allData.predictions[m]?.p50?.[h]) {
          row[m] = allData.predictions[m].p50[h];
        }
      });
      return row;
    });
  };

  const tabs = [
    { key: 'forecast',    label: 'Forecast' },
    { key: 'comparison',  label: 'Model Comparison' },
    { key: 'chart',       label: 'Price Chart' },
    { key: 'sentiment',   label: 'Sentiment' },
    { key: 'fundamentals',label: 'Fundamentals' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">

      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-wrap justify-between items-center py-5 gap-4">
            <div className="flex items-center">
              <FaChartLine className="h-7 w-7 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">Stock Prediction Dashboard</h1>
            </div>

            <div className="flex items-center gap-3 flex-wrap">
              {/* Model selector */}
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500"
              >
                <option value="transformer">Transformer</option>
                <option value="lstm">LSTM</option>
                <option value="rnn">RNN</option>
                <option value="rf">Random Forest</option>
              </select>

              {/* Ticker search */}
              <div className="relative">
                <input
                  type="text"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  onKeyPress={handleKeyPress}
                  placeholder="Ticker (e.g. AAPL)"
                  className="w-52 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-blue-600 disabled:opacity-50"
                >
                  {loading ? <FaSpinner className="animate-spin" /> : <FaSearch />}
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

        {/* Error banner */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
            {error}
          </div>
        )}

        {loading && <LoadingSpinner />}

        {data && !loading && (
          <ErrorBoundary>

            {/* Ticker and price bar */}
            <div className="bg-white rounded-lg shadow p-5 mb-6 flex justify-between items-center">
              <div>
                <h2 className="text-3xl font-bold text-gray-900">{data.ticker}</h2>
                <p className="text-gray-500 text-sm mt-1">
                  Current price: <span className="font-semibold text-gray-800">${data.current_price}</span>
                  &nbsp;&nbsp;|&nbsp;&nbsp;Model: <span className="font-semibold text-blue-600 capitalize">{data.model}</span>
                </p>
              </div>
              <p className="text-sm text-gray-400">{new Date().toLocaleDateString()}</p>
            </div>

            {/* Tab navigation */}
            <div className="flex gap-1 mb-6 border-b border-gray-200">
              {tabs.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setActiveTab(t.key)}
                  className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                    activeTab === t.key
                      ? 'bg-white border border-b-white border-gray-200 text-blue-600'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {t.label}
                </button>
              ))}
            </div>

            {/* Forecast tab */}
            {activeTab === 'forecast' && (
              <div className="space-y-6">
                <div className="bg-white rounded-lg shadow p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Price Predictions — {data.model}</h3>
                  <PredictionTable
                    predictions={data.predictions}
                    p10={data.p10}
                    p90={data.p90}
                    currentPrice={data.current_price}
                  />
                  {data.p10 && data.p90 && data.p10 !== data.p50 && (
                    <p className="text-xs text-gray-400 mt-3">
                      p10 / p90 bounds are Monte Carlo Dropout confidence intervals (50 forward passes).
                    </p>
                  )}
                </div>
                <div className="bg-white rounded-lg shadow p-6">
                  <div className="flex items-center mb-4">
                    <FaLightbulb className="h-5 w-5 text-yellow-500 mr-2" />
                    <h3 className="text-lg font-medium text-gray-900">Recommendation</h3>
                  </div>
                  <RecommendationCard recommendation={data.recommendation} />
                </div>
              </div>
            )}

            {/* Model comparison tab */}
            {activeTab === 'comparison' && allData && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-6">
                  All 4 Models — p50 Price Forecasts
                </h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={buildComparisonData(allData)} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="horizon" tick={{ fontSize: 12 }} />
                      <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v}`} />
                      <Tooltip formatter={(v) => `$${v.toFixed(2)}`} />
                      <Legend />
                      {MODELS.map((m) => (
                        <Bar key={m} dataKey={m} name={m.toUpperCase()} fill={MODEL_COLORS[m]} />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                {Object.keys(allData.errors || {}).length > 0 && (
                  <div className="mt-4 text-sm text-red-500">
                    {Object.entries(allData.errors).map(([m, e]) => (
                      <p key={m}>{m.toUpperCase()}: {e}</p>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Price chart tab */}
            {activeTab === 'chart' && (
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Price Chart with Moving Averages</h3>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={formatChartData(data.technical_data)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} angle={-35} textAnchor="end" height={70} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip
                        formatter={(v, n) => [`$${v.toFixed(2)}`, n]}
                        labelFormatter={(l) => `Date: ${l}`}
                        contentStyle={{ borderRadius: 8, fontSize: 12 }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="Close Price" stroke="#3b82f6" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="SMA 50"      stroke="#22c55e" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="SMA 200"     stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Sentiment tab */}
            {activeTab === 'sentiment' && (
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center mb-4">
                  <FaNewspaper className="h-5 w-5 text-blue-500 mr-2" />
                  <h3 className="text-lg font-medium text-gray-900">News Sentiment</h3>
                </div>
                <SentimentAnalysis sentimentData={sentiment} />
              </div>
            )}

            {/* Fundamentals tab */}
            {activeTab === 'fundamentals' && (
              <div className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center mb-4">
                  <FaBuilding className="h-5 w-5 text-purple-500 mr-2" />
                  <h3 className="text-lg font-medium text-gray-900">Company Fundamentals</h3>
                </div>
                <CompanyFundamentals fundamentals={data.fundamentals} />
              </div>
            )}

          </ErrorBoundary>
        )}

        {/* Empty state */}
        {!data && !loading && !error && (
          <div className="text-center py-20 text-gray-400">
            <FaChartLine className="mx-auto h-12 w-12 mb-4" />
            <p className="text-sm">Enter a ticker symbol to get started.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
