import openai
import os
from typing import List, Dict
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenAIRecommendationService:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI service with API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file or pass it to constructor.")
        
        openai.api_key = self.api_key
    
    def get_factor_recommendations(self, contributing_factors: List[str], customer_data: Dict) -> Dict:
        """
        Get AI-powered recommendations for each contributing factor.
        
        Args:
            contributing_factors: List of top 3 contributing factors
            customer_data: Customer data used for prediction
            
        Returns:
            Dictionary with recommendations for each factor
        """
        try:
            # Create context for the AI
            context = self._create_context(contributing_factors, customer_data)
            
            # Generate recommendations using OpenAI
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a customer retention expert. Analyze the contributing factors to customer churn and provide actionable, specific recommendations for each factor. 
                        
                        Format your response as a JSON object with the following structure:
                        {
                            "recommendations": {
                                "factor_name_1": {
                                    "analysis": "Brief analysis of why this factor contributes to churn",
                                    "recommendations": ["Specific action 1", "Specific action 2", "Specific action 3"],
                                    "priority": "high/medium/low"
                                },
                                "factor_name_2": {
                                    "analysis": "Brief analysis of why this factor contributes to churn", 
                                    "recommendations": ["Specific action 1", "Specific action 2", "Specific action 3"],
                                    "priority": "high/medium/low"
                                },
                                "factor_name_3": {
                                    "analysis": "Brief analysis of why this factor contributes to churn",
                                    "recommendations": ["Specific action 1", "Specific action 2", "Specific action 3"], 
                                    "priority": "high/medium/low"
                                }
                            },
                            "overall_strategy": "Brief overall retention strategy summary"
                        }
                        
                        Be specific, actionable, and business-focused. Consider the customer's current behavior patterns."""
                    },
                    {
                        "role": "user", 
                        "content": context
                    }
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the response
            content = response.choices[0].message.content
            recommendations = json.loads(content)
            
            return recommendations
            
        except Exception as e:
            # Fallback to basic recommendations if OpenAI fails
            return self._get_fallback_recommendations(contributing_factors)
    
    def _create_context(self, contributing_factors: List[str], customer_data: Dict) -> str:
        """Create context string for OpenAI prompt."""
        context = f"""
        Customer Data:
        - Days since last purchase: {customer_data.get('days_since_last_purchase', 'N/A')} days
        - Average orders per month: {customer_data.get('avg_orders_per_month', 'N/A')}
        - Total spent: ${customer_data.get('total_spent', 'N/A'):,.2f}
        - Average spent per order: ${customer_data.get('avg_spent_per_order', 'N/A'):,.2f}
        - Order duration: {customer_data.get('order_duration_months', 'N/A')} months
        
        Top 3 Contributing Factors to Churn Risk:
        {chr(10).join([f"{i+1}. {factor}" for i, factor in enumerate(contributing_factors)])}
        
        Please provide specific, actionable recommendations for each factor to reduce churn risk.
        """
        return context
    
    def _get_fallback_recommendations(self, contributing_factors: List[str]) -> Dict:
        """Fallback recommendations if OpenAI API fails."""
        fallback_recommendations = {
            "days_since_last_purchase": {
                "analysis": "Customer hasn't made a purchase recently, indicating declining engagement",
                "recommendations": [
                    "Send personalized re-engagement email campaigns",
                    "Offer limited-time discounts or promotions",
                    "Implement win-back campaigns with exclusive offers"
                ],
                "priority": "high"
            },
            "avg_orders_per_month": {
                "analysis": "Low order frequency suggests reduced customer engagement",
                "recommendations": [
                    "Create subscription or loyalty programs",
                    "Implement automated reminder systems",
                    "Offer incentives for regular purchases"
                ],
                "priority": "medium"
            },
            "total_spent": {
                "analysis": "Low total spend indicates limited customer value or engagement",
                "recommendations": [
                    "Upsell and cross-sell opportunities",
                    "Create premium product bundles",
                    "Implement tiered loyalty programs"
                ],
                "priority": "medium"
            }
        }
        
        # Filter to only include the actual contributing factors
        filtered_recommendations = {
            factor: fallback_recommendations.get(factor, {
                "analysis": f"Factor {factor} is contributing to churn risk",
                "recommendations": ["Analyze customer behavior patterns", "Implement targeted retention strategies"],
                "priority": "medium"
            })
            for factor in contributing_factors
        }
        
        return {
            "recommendations": filtered_recommendations,
            "overall_strategy": "Focus on re-engagement and value creation strategies"
        } 