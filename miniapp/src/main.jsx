import React, { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'

class RootErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, message: '' }
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      message: error?.message || 'Unknown error',
    }
  }

  componentDidCatch(error) {
    console.error('Root render error:', error)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className='mesh-bg min-h-screen grid place-items-center text-white'>
          <div className='glass-panel rounded-2xl px-6 py-5 text-center text-sm'>
            <div className='mb-2 text-base font-semibold'>Mini app xatolikka uchradi</div>
            <div className='mb-4 opacity-70'>{this.state.message}</div>
            <button className='btn-primary' onClick={() => window.location.reload()}>
              Qayta yuklash
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </StrictMode>,
)
