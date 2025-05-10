import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Suspense } from 'react'

import { ReactFlowProvider } from 'reactflow'

import Routing from './pages'
import { CommonMenu } from '@/shared/ui'
import { ChakraProvider, defaultSystem } from "@chakra-ui/react"

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <ChakraProvider value={defaultSystem}>
    <ReactFlowProvider>
      <BrowserRouter>
        <Suspense fallback={<CommonMenu />}>
          <Routing />
        </Suspense>
      </BrowserRouter>
    </ReactFlowProvider>
  </ChakraProvider>
)
