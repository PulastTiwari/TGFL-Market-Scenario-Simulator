'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, TrendingUp, Download, Sparkles, BarChart3 } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface ScenarioRequest {
  num_scenarios: number
  regime: 'normal' | 'bull' | 'bear' | 'volatile'
  length: number
  asset_symbol: string
  seed?: number
}

interface ScenarioResponse {
  scenario_id: string
  regime: string
  length: number
  asset_symbol: string
  data: number[][]
  timestamps: string[]
  metadata: {
    seed?: number
    model_version: string
    generation_time: string
  }
  created_at: string
}

export default function ScenariosPage() {
  const [scenarios, setScenarios] = useState<ScenarioResponse[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [selectedScenario, setSelectedScenario] = useState<ScenarioResponse | null>(null)
  
  // Generation configuration
  const [config, setConfig] = useState<ScenarioRequest>({
    num_scenarios: 5,
    regime: 'normal',
    length: 100,
    asset_symbol: 'SPY',
    seed: undefined
  })

  const generateScenarios = async () => {
    setIsGenerating(true)
    try {
      const response = await fetch('http://localhost:8000/scenarios/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      if (response.ok) {
        const data = await response.json()
        setScenarios(prev => [data, ...prev])
        setSelectedScenario(data)
      } else {
        console.error('Failed to generate scenarios')
      }
    } catch (error) {
      console.error('Error generating scenarios:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  // Prepare chart data for selected scenario
  const getChartData = (scenario: ScenarioResponse) => {
    if (!scenario.data[0]) return []
    
    return scenario.data[0].map((price, index) => ({
      day: index,
      price: price,
      time: scenario.timestamps[index]?.split('T')[0] || `Day ${index}`
    }))
  }

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'bull': return 'bg-green-100 text-green-800'
      case 'bear': return 'bg-red-100 text-red-800'
      case 'volatile': return 'bg-purple-100 text-purple-800'
      default: return 'bg-blue-100 text-blue-800'
    }
  }

  const getRegimeDescription = (regime: string) => {
    switch (regime) {
      case 'bull': return 'Upward trending market with positive returns'
      case 'bear': return 'Downward trending market with negative returns'
      case 'volatile': return 'High volatility market with large price swings'
      case 'normal': return 'Stable market with moderate volatility'
      default: return 'Market scenario'
    }
  }

  const calculateStatistics = (data: number[]) => {
    if (data.length === 0) return { returns: [], volatility: 0, maxDrawdown: 0, totalReturn: 0 }
    
    const returns = []
    for (let i = 1; i < data.length; i++) {
      returns.push((data[i] - data[i-1]) / data[i-1])
    }
    
    const volatility = Math.sqrt(returns.reduce((sum, r) => sum + r * r, 0) / returns.length) * Math.sqrt(252)
    const totalReturn = (data[data.length - 1] - data[0]) / data[0]
    
    // Calculate max drawdown
    let maxPrice = data[0]
    let maxDrawdown = 0
    for (const price of data) {
      if (price > maxPrice) maxPrice = price
      const drawdown = (maxPrice - price) / maxPrice
      if (drawdown > maxDrawdown) maxDrawdown = drawdown
    }
    
    return { returns, volatility, maxDrawdown, totalReturn }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Button variant="outline" size="icon" asChild>
          <Link href="/">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Market Scenarios</h1>
          <p className="text-muted-foreground">
            Generate realistic market scenarios using trained transformer models
          </p>
        </div>
      </div>

      {/* Configuration and Generation */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Generation Settings</CardTitle>
            <CardDescription>
              Configure scenario generation parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Market Regime</label>
              <select 
                className="w-full p-2 border rounded-md"
                value={config.regime}
                onChange={(e) => setConfig({...config, regime: e.target.value as any})}
              >
                <option value="normal">Normal Market</option>
                <option value="bull">Bull Market</option>
                <option value="bear">Bear Market</option>
                <option value="volatile">Volatile Market</option>
              </select>
              <p className="text-xs text-muted-foreground">
                {getRegimeDescription(config.regime)}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium">Scenarios</label>
                <select 
                  className="w-full p-2 border rounded-md"
                  value={config.num_scenarios}
                  onChange={(e) => setConfig({...config, num_scenarios: Number(e.target.value)})}
                >
                  {[1, 3, 5, 10, 20].map(n => (
                    <option key={n} value={n}>{n} path{n > 1 ? 's' : ''}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Length (days)</label>
                <select 
                  className="w-full p-2 border rounded-md"
                  value={config.length}
                  onChange={(e) => setConfig({...config, length: Number(e.target.value)})}
                >
                  {[50, 100, 252, 500].map(n => (
                    <option key={n} value={n}>{n} days</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Asset Symbol</label>
              <input 
                type="text"
                className="w-full p-2 border rounded-md"
                value={config.asset_symbol}
                onChange={(e) => setConfig({...config, asset_symbol: e.target.value})}
                placeholder="e.g., SPY, AAPL, BTC"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Random Seed (optional)</label>
              <input 
                type="number"
                className="w-full p-2 border rounded-md"
                value={config.seed || ''}
                onChange={(e) => setConfig({...config, seed: e.target.value ? Number(e.target.value) : undefined})}
                placeholder="Leave empty for random"
              />
            </div>

            <Button 
              className="w-full" 
              onClick={generateScenarios}
              disabled={isGenerating}
            >
              <Sparkles className="h-4 w-4 mr-2" />
              {isGenerating ? 'Generating...' : 'Generate Scenarios'}
            </Button>
          </CardContent>
        </Card>

        {/* Scenario Visualization */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Scenario Visualization</span>
              {selectedScenario && (
                <Badge className={getRegimeColor(selectedScenario.regime)}>
                  {selectedScenario.regime} market
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              {selectedScenario 
                ? `Price path for ${selectedScenario.asset_symbol} • ${selectedScenario.data.length} scenario${selectedScenario.data.length > 1 ? 's' : ''}`
                : 'Select or generate a scenario to view the price chart'
              }
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedScenario ? (
              <div className="space-y-4">
                {/* Chart */}
                <div className="h-64 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={getChartData(selectedScenario)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="day" 
                        type="number"
                        scale="linear"
                        domain={['dataMin', 'dataMax']}
                      />
                      <YAxis 
                        domain={['dataMin - 5', 'dataMax + 5']}
                        tickFormatter={(value) => `$${value.toFixed(0)}`}
                      />
                      <Tooltip 
                        formatter={(value: any) => [`$${Number(value).toFixed(2)}`, 'Price']}
                        labelFormatter={(label) => `Day ${label}`}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="price" 
                        stroke="#2563eb" 
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Statistics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
                  {(() => {
                    const stats = calculateStatistics(selectedScenario.data[0] || [])
                    return (
                      <>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Total Return</p>
                          <p className={`text-lg font-bold ${stats.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {(stats.totalReturn * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Annualized Volatility</p>
                          <p className="text-lg font-bold">
                            {(stats.volatility * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Max Drawdown</p>
                          <p className="text-lg font-bold text-red-600">
                            -{(stats.maxDrawdown * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground">Days</p>
                          <p className="text-lg font-bold">
                            {selectedScenario.length}
                          </p>
                        </div>
                      </>
                    )
                  })()}
                </div>

                {/* Actions */}
                <div className="flex gap-2 pt-4 border-t">
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4 mr-2" />
                    Export CSV
                  </Button>
                  <Button variant="outline" size="sm">
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Detailed Analysis
                  </Button>
                </div>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <TrendingUp className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <p>No scenarios selected</p>
                  <p className="text-sm mt-1">Generate scenarios to view price charts and statistics</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Generated Scenarios */}
      <Card>
        <CardHeader>
          <CardTitle>Generated Scenarios</CardTitle>
          <CardDescription>
            History of generated market scenarios and their properties
          </CardDescription>
        </CardHeader>
        <CardContent>
          {scenarios.length > 0 ? (
            <div className="space-y-3">
              {scenarios.map((scenario) => {
                const stats = calculateStatistics(scenario.data[0] || [])
                return (
                  <div 
                    key={scenario.scenario_id} 
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedScenario?.scenario_id === scenario.scenario_id 
                        ? 'border-blue-500 bg-blue-50' 
                        : 'hover:bg-gray-50'
                    }`}
                    onClick={() => setSelectedScenario(scenario)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <Badge className={getRegimeColor(scenario.regime)}>
                          {scenario.regime}
                        </Badge>
                        <div>
                          <p className="font-medium">
                            {scenario.asset_symbol} • {scenario.data.length} scenario{scenario.data.length > 1 ? 's' : ''}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {scenario.length} days • Generated {new Date(scenario.created_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <div className="text-right">
                          <p className="text-muted-foreground">Return</p>
                          <p className={stats.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'}>
                            {(stats.totalReturn * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div className="text-right">
                          <p className="text-muted-foreground">Volatility</p>
                          <p>{(stats.volatility * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="text-center py-8">
              <Sparkles className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <p className="text-muted-foreground">No scenarios generated yet</p>
              <p className="text-xs text-muted-foreground mt-1">Configure parameters and generate your first market scenario</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}