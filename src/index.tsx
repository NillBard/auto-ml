import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Suspense } from 'react'

import { ReactFlowProvider } from 'reactflow'

import Routing from './pages'
import { CommonMenu } from '@/shared/ui'

import { Provider } from '@/shared/libs'

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <Provider>
    <ReactFlowProvider>
      <BrowserRouter>
        <Suspense fallback={<CommonMenu />}>
          <Routing />
        </Suspense>
      </BrowserRouter>
    </ReactFlowProvider>
  </Provider>
)
