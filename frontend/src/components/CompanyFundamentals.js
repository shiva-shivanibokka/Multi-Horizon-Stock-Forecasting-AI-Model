import React, { useState } from 'react';
import { FaChevronDown, FaChevronRight, FaBuilding, FaChartLine, FaDollarSign, FaGift, FaArrowUp, FaInfoCircle } from 'react-icons/fa';

const CompanyFundamentals = ({ fundamentals }) => {
    const [expandedSections, setExpandedSections] = useState({
        company_info: true,
        valuation_metrics: false,
        financial_metrics: false,
        dividend_info: false,
        growth_metrics: false,
        trading_info: false
    });

    const toggleSection = (section) => {
        setExpandedSections(prev => ({
            ...prev,
            [section]: !prev[section]
        }));
    };

    const renderMetric = (label, value, format = 'text') => {
        if (value === 'N/A' || value === null || value === undefined) {
            return <span className="text-gray-400">N/A</span>;
        }

        switch (format) {
            case 'currency':
                return <span className="font-semibold text-green-600">{value}</span>;
            case 'percentage':
                const numValue = parseFloat(value);
                const color = numValue > 0 ? 'text-green-600' : numValue < 0 ? 'text-red-600' : 'text-gray-600';
                return <span className={`font-semibold ${color}`}>{value}%</span>;
            case 'ratio':
                return <span className="font-semibold text-blue-600">{value}</span>;
            default:
                return <span className="font-semibold text-gray-800">{value}</span>;
        }
    };

    const renderSection = (title, icon, data, sectionKey) => {
        if (!data || Object.keys(data).length === 0) return null;

        return (
            <div className="bg-white rounded-lg shadow-md mb-4">
                <button
                    onClick={() => toggleSection(sectionKey)}
                    className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-gray-50 transition-colors"
                >
                    <div className="flex items-center space-x-2">
                        {icon}
                        <span className="font-semibold text-gray-800">{title}</span>
                    </div>
                    {expandedSections[sectionKey] ? <FaChevronDown /> : <FaChevronRight />}
                </button>
                
                {expandedSections[sectionKey] && (
                    <div className="px-4 pb-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {Object.entries(data).map(([key, value]) => {
                                if (key === 'error') return null;
                                
                                const label = key.split('_').map(word => 
                                    word.charAt(0).toUpperCase() + word.slice(1)
                                ).join(' ');
                                
                                let format = 'text';
                                if (key.includes('yield') || key.includes('growth') || key.includes('margins') || 
                                    key.includes('change') || key.includes('ratio')) {
                                    format = 'percentage';
                                } else if (key.includes('revenue') || key.includes('income') || key.includes('profit') || 
                                         key.includes('cash') || key.includes('debt') || key.includes('cap') || 
                                         key.includes('value') || key.includes('rate')) {
                                    format = 'currency';
                                } else if (key.includes('pe_') || key.includes('price_to_') || key.includes('enterprise_to_') || 
                                         key.includes('debt_to_') || key.includes('current_') || key.includes('quick_') || 
                                         key.includes('beta')) {
                                    format = 'ratio';
                                }
                                
                                return (
                                    <div key={key} className="flex justify-between items-center py-2 border-b border-gray-100">
                                        <span className="text-sm text-gray-600">{label}</span>
                                        {renderMetric(label, value, format)}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    if (!fundamentals || fundamentals.error) {
        return (
            <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center space-x-2 mb-4">
                    <FaInfoCircle className="text-blue-500" />
                    <h3 className="text-lg font-semibold text-gray-800">Company Fundamentals</h3>
                </div>
                <p className="text-gray-500">
                    {fundamentals?.error || 'Fundamental data not available for this ticker.'}
                </p>
            </div>
        );
    }

        return (
        <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center space-x-2 mb-6">
                <FaBuilding className="text-blue-500" />
                <h3 className="text-lg font-semibold text-gray-800">Company Fundamentals</h3>
            </div>
            
            {/* Top 10 Fundamentals with Values */}
            <div className="space-y-4 mb-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* P/E Ratio */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">P/E Ratio</span>
                            <span className="font-semibold text-blue-600">
                                {fundamentals?.valuation_metrics?.pe_ratio || 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Price-to-Earnings ratio indicates valuation relative to earnings</p>
                    </div>

                    {/* Market Cap */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Market Cap</span>
                            <span className="font-semibold text-green-600">
                                {fundamentals?.company_info?.market_cap || 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Total value of company's shares</p>
                    </div>

                    {/* Revenue */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Revenue</span>
                            <span className="font-semibold text-green-600">
                                {fundamentals?.financial_metrics?.revenue || 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Total sales and business income</p>
                    </div>

                    {/* Net Income */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Net Income</span>
                            <span className="font-semibold text-green-600">
                                {fundamentals?.financial_metrics?.net_income || 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Profit after all expenses</p>
                    </div>

                    {/* Debt-to-Equity */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Debt-to-Equity</span>
                            <span className="font-semibold text-blue-600">
                                {fundamentals?.financial_metrics?.debt_to_equity || 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Financial leverage ratio</p>
                    </div>

                    {/* Beta */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Beta</span>
                            <span className="font-semibold text-blue-600">
                                {fundamentals?.trading_info?.beta || 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Stock volatility compared to market (1.0 = market average)</p>
                    </div>

                    {/* Dividend Yield */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Dividend Yield</span>
                            <span className="font-semibold text-green-600">
                                {fundamentals?.dividend_info?.dividend_yield ? `${fundamentals.dividend_info.dividend_yield}%` : 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Annual dividend as percentage of stock price</p>
                    </div>

                    {/* Revenue Growth */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Revenue Growth</span>
                            <span className="font-semibold text-green-600">
                                {fundamentals?.growth_metrics?.revenue_growth ? `${fundamentals.growth_metrics.revenue_growth}%` : 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Year-over-year sales growth</p>
                    </div>

                    {/* Profit Margins */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">Profit Margins</span>
                            <span className="font-semibold text-green-600">
                                {fundamentals?.growth_metrics?.profit_margins ? `${fundamentals.growth_metrics.profit_margins}%` : 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Net income as percentage of revenue</p>
                    </div>

                    {/* 52-Week Change */}
                    <div className="bg-gray-50 p-4 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-sm font-medium text-gray-700">52-Week Change</span>
                            <span className="font-semibold text-green-600">
                                {fundamentals?.trading_info?.fifty_two_week_change ? `${fundamentals.trading_info.fifty_two_week_change}%` : 'N/A'}
                            </span>
                        </div>
                        <p className="text-xs text-gray-600">Stock performance over the past year</p>
                    </div>
                </div>
            </div>


        </div>
    );
};

export default CompanyFundamentals; 