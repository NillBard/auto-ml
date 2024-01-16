import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Suspense } from 'react'

import { ChakraProvider } from '@chakra-ui/react'
import { ReactFlowProvider } from 'reactflow'

import Routing from './pages'

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <ChakraProvider>
    <ReactFlowProvider>
      <BrowserRouter>
        <Suspense fallback={<div>loading</div>}>
          <Routing />
        </Suspense>
      </BrowserRouter>
    </ReactFlowProvider>
  </ChakraProvider>
)
