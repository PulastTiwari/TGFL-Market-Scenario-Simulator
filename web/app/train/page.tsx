'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Play, Square, Users, Activity, Clock, TrendingUp } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

interface FederatedTrainingRequest {
  num_clients: number
  num_rounds: number
  total_samples: number
  partition_strategy: 'iid' | 'non_iid' | 'temporal'
  server_address: string
  wait_for_completion: boolean
}

interface FederatedTrainingStatus {
  simulation_id: string
  status: 'pending' | 'preparing' | 'running' | 'completed' | 'failed' | 'stopped'
  current_round?: number
  total_rounds: number
  progress: number
  start_time?: string
  end_time?: string
  simulation_time?: number
  server_running: boolean
  active_clients: number
  total_clients: number
  error_message?: string
  results_path?: string
}

interface SimulationSummary {
  simulation_id: string
  status: string
  total_rounds: number
  progress: number
  start_time?: string
  end_time?: string
}

export default function FederatedTrainingPage() {
  const [currentSimulation, setCurrentSimulation] = useState<FederatedTrainingStatus | null>(null)
  const [simulations, setSimulations] = useState<SimulationSummary[]>([])
  const [isStarting, setIsStarting] = useState(false)
  const [loading, setLoading] = useState(true)

  // Training configuration
  const [config, setConfig] = useState<FederatedTrainingRequest>({
    num_clients: 3,
    num_rounds: 5,
    total_samples: 300,
    partition_strategy: 'iid',
    server_address: 'localhost:8080',
    wait_for_completion: true
  })

  useEffect(() => {
    fetchSimulations()
    const interval = setInterval(fetchSimulations, 5000) // Poll every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchSimulations = async () => {
    try {
      const response = await fetch('http://localhost:8000/federated/simulations')
      const data = await response.json()
      setSimulations(data.simulations || [])
      
      // Find active simulation
      const active = data.simulations?.find((sim: SimulationSummary) => 
        ['pending', 'preparing', 'running'].includes(sim.status)
      )
      
      if (active && (!currentSimulation || currentSimulation.simulation_id !== active.simulation_id)) {
        fetchSimulationStatus(active.simulation_id)
      }
    } catch (error) {
      console.error('Failed to fetch simulations:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchSimulationStatus = async (simulationId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/federated/train/status/${simulationId}`)
      const data = await response.json()
      setCurrentSimulation(data)
    } catch (error) {
      console.error('Failed to fetch simulation status:', error)
    }
  }

  const startTraining = async () => {
    setIsStarting(true)
    try {
      const response = await fetch('http://localhost:8000/federated/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      })
      
      if (response.ok) {
        const data = await response.json()
        setCurrentSimulation(data)
        // Start polling for updates
        setTimeout(() => fetchSimulations(), 1000)
      } else {
        console.error('Failed to start training')
      }
    } catch (error) {
      console.error('Error starting training:', error)
    } finally {
      setIsStarting(false)
    }
  }

  const stopTraining = async () => {
    if (!currentSimulation) return
    
    try {
      const response = await fetch(`http://localhost:8000/federated/train/stop/${currentSimulation.simulation_id}`, {
        method: 'POST'
      })
      
      if (response.ok) {
        setCurrentSimulation(null)
        fetchSimulations()
      }
    } catch (error) {
      console.error('Error stopping training:', error)
    }
  }

  const runQuickTest = async () => {
    setIsStarting(true)
    try {
      const response = await fetch('http://localhost:8000/federated/train/quick')
      const data = await response.json()
      
      if (data.success) {
        alert(`Quick test completed in ${data.simulation_time?.toFixed(1)}s`)
      } else {
        alert(`Quick test failed: ${data.error}`)
      }
    } catch (error) {
      console.error('Error running quick test:', error)
    } finally {
      setIsStarting(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800'
      case 'running': case 'preparing': return 'bg-blue-100 text-blue-800'
      case 'failed': return 'bg-red-100 text-red-800'
      case 'stopped': return 'bg-gray-100 text-gray-800'
      default: return 'bg-yellow-100 text-yellow-800'
    }
  }

  const formatTime = (timeStr?: string) => {
    if (!timeStr) return 'N/A'
    return new Date(timeStr).toLocaleTimeString()
  }

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A'
    if (seconds < 60) return `${seconds.toFixed(0)}s`
    return `${(seconds / 60).toFixed(1)}m`
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
          <h1 className="text-3xl font-bold tracking-tight">Federated Training</h1>
          <p className="text-muted-foreground">
            Configure and run federated learning simulations with distributed clients
          </p>
        </div>
      </div>

      {/* Current Simulation Status */}
      {currentSimulation && (
        <Card className="border-blue-200 bg-blue-50">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Active Simulation
              </span>
              <Badge className={getStatusColor(currentSimulation.status)}>
                {currentSimulation.status.charAt(0).toUpperCase() + currentSimulation.status.slice(1)}
              </Badge>
            </CardTitle>
            <CardDescription>
              Simulation ID: {currentSimulation.simulation_id}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{Math.round(currentSimulation.progress * 100)}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500" 
                  style={{ width: `${currentSimulation.progress * 100}%` }}
                />
              </div>
              {currentSimulation.current_round && (
                <p className="text-xs text-muted-foreground">
                  Round {currentSimulation.current_round} of {currentSimulation.total_rounds}
                </p>
              )}
            </div>

            {/* Status Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="space-y-1">
                <p className="text-sm font-medium">Server Status</p>
                <p className="text-2xl font-bold">
                  {currentSimulation.server_running ? '🟢' : '🔴'}
                </p>
                <p className="text-xs text-muted-foreground">
                  {currentSimulation.server_running ? 'Running' : 'Stopped'}
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Active Clients</p>
                <p className="text-2xl font-bold">
                  {currentSimulation.active_clients}/{currentSimulation.total_clients}
                </p>
                <p className="text-xs text-muted-foreground">Connected</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Duration</p>
                <p className="text-2xl font-bold">
                  {formatDuration(currentSimulation.simulation_time)}
                </p>
                <p className="text-xs text-muted-foreground">Running time</p>
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Start Time</p>
                <p className="text-lg font-bold">
                  {formatTime(currentSimulation.start_time)}
                </p>
                <p className="text-xs text-muted-foreground">Started at</p>
              </div>
            </div>

            {/* Error Message */}
            {currentSimulation.error_message && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-800">{currentSimulation.error_message}</p>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              <Button 
                variant="destructive" 
                onClick={stopTraining}
                disabled={!['running', 'preparing'].includes(currentSimulation.status)}
              >
                <Square className="h-4 w-4 mr-2" />
                Stop Simulation
              </Button>
              {currentSimulation.status === 'completed' && currentSimulation.results_path && (
                <Button variant="outline">
                  View Results
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Configuration & Start */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Training Configuration</CardTitle>
            <CardDescription>
              Configure the federated learning simulation parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Clients</label>
                <select 
                  className="w-full p-2 border rounded-md"
                  value={config.num_clients}
                  onChange={(e) => setConfig({...config, num_clients: Number(e.target.value)})}
                >
                  {[2, 3, 4, 5].map(n => (
                    <option key={n} value={n}>{n} clients</option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Rounds</label>
                <select 
                  className="w-full p-2 border rounded-md"
                  value={config.num_rounds}
                  onChange={(e) => setConfig({...config, num_rounds: Number(e.target.value)})}
                >
                  {[3, 5, 10, 15, 20].map(n => (
                    <option key={n} value={n}>{n} rounds</option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Total Samples</label>
                <select 
                  className="w-full p-2 border rounded-md"
                  value={config.total_samples}
                  onChange={(e) => setConfig({...config, total_samples: Number(e.target.value)})}
                >
                  {[100, 300, 500, 1000].map(n => (
                    <option key={n} value={n}>{n} samples</option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Data Strategy</label>
                <select 
                  className="w-full p-2 border rounded-md"
                  value={config.partition_strategy}
                  onChange={(e) => setConfig({...config, partition_strategy: e.target.value as any})}
                >
                  <option value="iid">IID (Uniform)</option>
                  <option value="non_iid">Non-IID (Skewed)</option>
                  <option value="temporal">Temporal Split</option>
                </select>
              </div>
            </div>

            <div className="pt-4 border-t space-y-3">
              <Button 
                className="w-full" 
                onClick={startTraining}
                disabled={isStarting || (currentSimulation ? ['running', 'preparing'].includes(currentSimulation.status) : false)}
              >
                <Play className="h-4 w-4 mr-2" />
                {isStarting ? 'Starting...' : 'Start Federated Training'}
              </Button>
              
              <Button 
                variant="outline" 
                className="w-full" 
                onClick={runQuickTest}
                disabled={isStarting}
              >
                <Clock className="h-4 w-4 mr-2" />
                Run Quick Test (2 clients, 3 rounds)
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Recent Simulations */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Simulations</CardTitle>
            <CardDescription>
              History of federated training runs
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
              </div>
            ) : simulations.length > 0 ? (
              <div className="space-y-3">
                {simulations.slice(0, 5).map((sim) => (
                  <div key={sim.simulation_id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className={`h-2 w-2 rounded-full ${
                        sim.status === 'completed' ? 'bg-green-500' :
                        sim.status === 'failed' ? 'bg-red-500' :
                        sim.status === 'running' ? 'bg-blue-500' : 'bg-gray-500'
                      }`}></div>
                      <div>
                        <p className="text-sm font-medium">
                          {sim.total_rounds} rounds • {sim.simulation_id.slice(0, 8)}...
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {sim.start_time ? formatTime(sim.start_time) : 'Unknown time'}
                        </p>
                      </div>
                    </div>
                    <Badge variant="outline" className={getStatusColor(sim.status)}>
                      {sim.status}
                    </Badge>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <Users className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No simulations yet</p>
                <p className="text-xs text-muted-foreground mt-1">Start your first federated training</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}