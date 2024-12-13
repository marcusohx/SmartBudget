"use client"

import { useState } from "react"
import { Upload } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import ExpenseSummary from "./expense-summary"
import ExpenseCharts from "./expense-charts"

const months = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December"
]

const currentYear = new Date().getFullYear()
const years = Array.from({length: 5}, (_, i) => currentYear - i)

export default function ExpenseDashboard() {
  const [file, setFile] = useState<File | null>(null)
  const [month, setMonth] = useState<string>("")
  const [year, setYear] = useState<string>("")

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = () => {
    if (file && month && year) {
      // Here you would typically send the file, month, and year to your backend
      console.log("Uploading file:", file.name, "for", month, year)
      // For demo purposes, we'll just clear the inputs
      setFile(null)
      setMonth("")
      setYear("")
    }
  }

  return (
    <div className="container mx-auto p-4 space-y-6">
      <h1 className="text-3xl font-bold">Expense Dashboard</h1>
      
      <Card>
        <CardHeader>
          <CardTitle>Upload Expense PDF</CardTitle>
          <CardDescription>Upload your expense report PDF and specify the month and year</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4">
            <div>
              <Label htmlFor="pdf-upload">Expense PDF</Label>
              <Input
                id="pdf-upload"
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <Label htmlFor="month-select">Month</Label>
                <Select value={month} onValueChange={setMonth}>
                  <SelectTrigger id="month-select">
                    <SelectValue placeholder="Select month" />
                  </SelectTrigger>
                  <SelectContent>
                    {months.map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="year-select">Year</Label>
                <Select value={year} onValueChange={setYear}>
                  <SelectTrigger id="year-select">
                    <SelectValue placeholder="Select year" />
                  </SelectTrigger>
                  <SelectContent>
                    {years.map((y) => (
                      <SelectItem key={y} value={y.toString()}>{y}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <Button onClick={handleUpload} disabled={!file || !month || !year}>
              <Upload className="mr-2 h-4 w-4" /> Upload
            </Button>
          </div>
        </CardContent>
      </Card>
      
      <ExpenseSummary />
      <ExpenseCharts />
    </div>
  )
}

