'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Activity, Brain, TrendingUp, Users, Play, Settings, BarChart3 } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

interface SystemStatus {
  status: 'healthy' | 'degraded' | 'down'
  message: string
  uptime: number
  version: string
}

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/health')
        const data = await response.json()
        setSystemStatus({
          status: data.status === 'healthy' ? 'healthy' : 'degraded',
          message: data.message,
          uptime: data.uptime_seconds,
          version: data.version
        })
      } catch (error) {
        setSystemStatus({
          status: 'down',
          message: 'Unable to connect to API server',
          uptime: 0,
          version: 'unknown'
        })
      } finally {
        setLoading(false)
      }
    }

    fetchSystemStatus()
    const interval = setInterval(fetchSystemStatus, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-100 text-green-800'
      case 'degraded': return 'bg-yellow-100 text-yellow-800'
      case 'down': return 'bg-red-100 text-red-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const formatUptime = (seconds: number) => {
    if (seconds < 60) return `${Math.floor(seconds)}s`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">TGFL Market Scenario Simulator</h1>
          <p className="text-muted-foreground mt-2">
            Transformer-Based Generative Federated Learning for Financial Market Scenarios
          </p>
        </div>
        <div className="flex items-center gap-2">
          {loading ? (
            <Badge variant="outline">Checking...</Badge>
          ) : systemStatus ? (
            <Badge className={getStatusColor(systemStatus.status)}>
              {systemStatus.status.charAt(0).toUpperCase() + systemStatus.status.slice(1)}
            </Badge>
          ) : null}
        </div>
      </div>

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            System Status
          </CardTitle>
          <CardDescription>
            Real-time monitoring of API backend and federated learning components
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : systemStatus ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <p className="text-sm font-medium">API Status</p>
                <p className="text-2xl font-bold">{systemStatus.status}</p>
                <p className="text-xs text-muted-foreground">{systemStatus.message}</p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Uptime</p>
                <p className="text-2xl font-bold">{formatUptime(systemStatus.uptime)}</p>
                <p className="text-xs text-muted-foreground">Server running time</p>
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Version</p>
                <p className="text-2xl font-bold">{systemStatus.version}</p>
                <p className="text-xs text-muted-foreground">API version</p>
              </div>
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-muted-foreground">Unable to fetch system status</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Federated Training */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5 text-blue-600" />
              Federated Training
            </CardTitle>
            <CardDescription>
              Train transformer models across distributed clients while preserving data privacy
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="font-medium">Model Architecture</p>
                <p className="text-muted-foreground">Tiny Transformer</p>
              </div>
              <div>
                <p className="font-medium">Parameters</p>
                <p className="text-muted-foreground">&lt; 1M params</p>
              </div>
              <div>
                <p className="font-medium">Training Method</p>
                <p className="text-muted-foreground">FedAvg Strategy</p>
              </div>
              <div>
                <p className="font-medium">Privacy</p>
                <p className="text-muted-foreground">Local Data Only</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button asChild className="flex-1">
                <Link href="/train">
                  <Play className="h-4 w-4 mr-2" />
                  Start Training
                </Link>
              </Button>
              <Button variant="outline" asChild>
                <Link href="/train">
                  <Settings className="h-4 w-4 mr-2" />
                  Configure
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Scenario Generation */}
        <Card className="hover:shadow-lg transition-shadow">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-green-600" />
              Market Scenarios
            </CardTitle>
            <CardDescription>
              Generate realistic market scenarios for risk analysis and stress testing
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="font-medium">Market Regimes</p>
                <p className="text-muted-foreground">Bull, Bear, Normal, Volatile</p>
              </div>
              <div>
                <p className="font-medium">Validation</p>
                <p className="text-muted-foreground">KS Test, ACF Analysis</p>
              </div>
              <div>
                <p className="font-medium">Time Horizon</p>
                <p className="text-muted-foreground">Configurable length</p>
              </div>
              <div>
                <p className="font-medium">Output Format</p>
                <p className="text-muted-foreground">Price paths, Returns</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Button asChild className="flex-1">
                <Link href="/scenarios">
                  <Brain className="h-4 w-4 mr-2" />
                  Generate Scenarios
                </Link>
              </Button>
              <Button variant="outline" asChild>
                <Link href="/scenarios">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  View Results
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>
            Latest federated training runs and scenario generation jobs
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 rounded-full bg-green-500"></div>
                <div>
                  <p className="text-sm font-medium">Federated Training - 4 clients</p>
                  <p className="text-xs text-muted-foreground">Completed 2 minutes ago</p>
                </div>
              </div>
              <Badge variant="outline">Completed</Badge>
            </div>
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 rounded-full bg-blue-500"></div>
                <div>
                  <p className="text-sm font-medium">Bull Market Scenarios - 10 paths</p>
                  <p className="text-xs text-muted-foreground">Generated 15 minutes ago</p>
                </div>
              </div>
              <Badge variant="outline">Success</Badge>
            </div>
            <div className="flex items-center justify-between p-3 border rounded-lg">
              <div className="flex items-center gap-3">
                <div className="h-2 w-2 rounded-full bg-yellow-500"></div>
                <div>
                  <p className="text-sm font-medium">Federated Training - 2 clients</p>
                  <p className="text-xs text-muted-foreground">Started 1 hour ago</p>
                </div>
              </div>
              <Badge variant="outline">Running</Badge>
            </div>
          </div>
          <div className="mt-4 pt-4 border-t">
            <Button variant="outline" className="w-full">
              View All Activity
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}