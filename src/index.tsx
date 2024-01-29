import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { Suspense } from 'react'

import { ChakraProvider } from '@chakra-ui/react'
import { ReactFlowProvider } from 'reactflow'

import Routing from './pages'
import Menu from './components/commonMenu'

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <ChakraProvider>
    <ReactFlowProvider>
      <BrowserRouter>
        <Suspense fallback={<Menu />}>
          <Routing />
        </Suspense>
      </BrowserRouter>
    </ReactFlowProvider>
  </ChakraProvider>
)
