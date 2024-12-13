import { Metadata } from "next"
import ExpenseDashboard from "@/components/expense-dashboard"

export const metadata: Metadata = {
  title: "Expense Dashboard",
  description: "Upload your expense PDFs and analyze your spending",
}

export default function DashboardPage() {
  return <ExpenseDashboard />
}

