"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
} from "recharts";

const COLORS = [
  "#0088FE",
  "#00C49F",
  "#FFBB28",
  "#FF8042",
  "#8884D8",
  "#82ca9d",
];

const expenseData = [
  { month: "Jan", expenses: 4000, income: 5000, savings: 1000 },
  { month: "Feb", expenses: 3000, income: 5200, savings: 2200 },
  { month: "Mar", expenses: 2000, income: 4800, savings: 2800 },
  { month: "Apr", expenses: 2780, income: 5100, savings: 2320 },
  { month: "May", expenses: 1890, income: 5300, savings: 3410 },
  { month: "Jun", expenses: 2390, income: 5400, savings: 3010 },
  { month: "Jul", expenses: 3490, income: 5600, savings: 2110 },
];

const categoryData = [
  { name: "Housing", value: 35 },
  { name: "Food", value: 20 },
  { name: "Transportation", value: 15 },
  { name: "Utilities", value: 10 },
  { name: "Entertainment", value: 10 },
  { name: "Other", value: 10 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-background p-4 border rounded-lg shadow-lg">
        <p className="font-bold">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }}>
            {entry.name}: ${entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const CustomPieLabel = ({
  cx,
  cy,
  midAngle,
  innerRadius,
  outerRadius,
  percent,
  index,
}: any) => {
  const RADIAN = Math.PI / 180;
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return (
    <text
      x={x}
      y={y}
      fill="white"
      textAnchor={x > cx ? "start" : "end"}
      dominantBaseline="central"
      className="text-xs font-medium"
    >
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
};

export default function ExpenseCharts() {
  const [timeRange, setTimeRange] = useState("6months");

  const filteredData = expenseData.slice(
    -Number(timeRange.replace("months", ""))
  );

  const totalExpenses = filteredData.reduce(
    (sum, item) => sum + item.expenses,
    0
  );
  const totalIncome = filteredData.reduce((sum, item) => sum + item.income, 0);
  const totalSavings = filteredData.reduce(
    (sum, item) => sum + item.savings,
    0
  );
  const averageExpenses = totalExpenses / filteredData.length;
  const averageSavings = totalSavings / filteredData.length;
  const savingsRate = (totalSavings / totalIncome) * 100;

  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select time range" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="3months">Last 3 months</SelectItem>
            <SelectItem value="6months">Last 6 months</SelectItem>
            <SelectItem value="12months">Last 12 months</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Expense vs Income Trend</CardTitle>
            <CardDescription>
              Compare your expenses and income over time
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={filteredData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                >
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="expenses"
                    stroke="hsl(var(--primary))"
                  />
                  <Line
                    type="monotone"
                    dataKey="income"
                    stroke="hsl(var(--secondary))"
                  />
                  <Line
                    type="monotone"
                    dataKey="savings"
                    stroke="hsl(var(--accent))"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Expense Distribution</CardTitle>
            <CardDescription>
              Breakdown of your expenses by category
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={categoryData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={CustomPieLabel}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {categoryData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[index % COLORS.length]}
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Financial Insights</CardTitle>
          <CardDescription>
            Key metrics and trends from your financial data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div>
              <h3 className="font-semibold">Total Expenses</h3>
              <p className="text-2xl font-bold">${totalExpenses.toFixed(2)}</p>
            </div>
            <div>
              <h3 className="font-semibold">Average Monthly Expenses</h3>
              <p className="text-2xl font-bold">
                ${averageExpenses.toFixed(2)}
              </p>
            </div>
            <div>
              <h3 className="font-semibold">Total Savings</h3>
              <p className="text-2xl font-bold">${totalSavings.toFixed(2)}</p>
            </div>
            <div>
              <h3 className="font-semibold">Savings Rate</h3>
              <p className="text-2xl font-bold">{savingsRate.toFixed(2)}%</p>
            </div>
          </div>
          <div className="mt-6">
            <h3 className="font-semibold mb-2">Insights:</h3>
            <ul className="list-disc pl-5 space-y-2">
              <li>
                Your highest expense category is Housing at 35% of total
                expenses.
              </li>
              <li>
                Your average monthly savings is ${averageSavings.toFixed(2)}.
              </li>
              <li>
                Your savings rate of {savingsRate.toFixed(2)}% is{" "}
                {savingsRate > 20 ? "good" : "below the recommended 20%"}.
              </li>
              <li>
                {averageExpenses > 3000
                  ? "Your monthly expenses are high. Consider areas where you can cut back."
                  : "Your monthly expenses are within a reasonable range."}
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
