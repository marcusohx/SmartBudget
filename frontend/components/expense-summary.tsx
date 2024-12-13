import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { DollarSign, CreditCard, TrendingUp, TrendingDown } from 'lucide-react'

const summaryItems = [
  {
    title: "Total Spent",
    value: "$12,345",
    icon: DollarSign,
    trend: "up",
    trendValue: "10%",
  },
  {
    title: "Largest Expense",
    value: "$2,000",
    icon: CreditCard,
    description: "Rent",
  },
  {
    title: "Most Spending",
    value: "Groceries",
    icon: TrendingUp,
    trendValue: "$1,500",
  },
  {
    title: "Least Spending",
    value: "Entertainment",
    icon: TrendingDown,
    trendValue: "$100",
  },
]

export default function ExpenseSummary() {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {summaryItems.map((item, index) => (
        <Card key={index}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">{item.title}</CardTitle>
            <item.icon className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{item.value}</div>
            {item.trend && (
              <p className={`text-xs ${item.trend === 'up' ? 'text-green-500' : 'text-red-500'}`}>
                {item.trendValue} from last month
              </p>
            )}
            {item.description && (
              <p className="text-xs text-muted-foreground">{item.description}</p>
            )}
            {item.trendValue && !item.trend && (
              <p className="text-xs text-muted-foreground">{item.trendValue}</p>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

