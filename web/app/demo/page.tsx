'use client'

import { useState } from 'react'

export default function DemoPage() {
  const [runId, setRunId] = useState<string | null>(null)
  const [results, setResults] = useState<any | null>(null)
  const [loading, setLoading] = useState(false)

  const startTraining = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_clients: 2, num_rounds: 3 })
      })
      const data = await res.json()
      setRunId(data.run_id)
      pollResults(data.run_id)
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  const generateScenario = async () => {
    setLoading(true)
    try {
      const res = await fetch('http://localhost:8000/scenarios/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ regime: 'normal', length: 50, num_scenarios: 1 })
      })
      const data = await res.json()
      setRunId(data.scenario_id)
      pollResults(data.scenario_id)
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  const pollResults = async (id: string) => {
    try {
      const res = await fetch(`http://localhost:8000/results/${id}`)
      if (res.status === 200) {
        const data = await res.json()
        setResults(data)
        setLoading(false)
        return
      }
    } catch (err) {
      // ignore until available
    }

    setTimeout(() => pollResults(id), 1000)
  }

  return (
    <div className="p-6">
      <h2 className="text-2xl font-semibold mb-4">Demo</h2>
      <div className="flex gap-2">
        <button onClick={startTraining} className="px-4 py-2 bg-blue-600 text-white rounded">Start Training</button>
        <button onClick={generateScenario} className="px-4 py-2 bg-green-600 text-white rounded">Generate Scenario</button>
      </div>

      {loading && <p className="mt-4">Waiting for results...</p>}

      {runId && (
        <div className="mt-4">
          <p className="text-sm text-muted-foreground">Run ID: {runId}</p>
          <pre className="mt-2 p-2 bg-gray-100 rounded">{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}
