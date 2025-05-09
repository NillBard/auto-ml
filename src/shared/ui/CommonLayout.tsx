import { Box } from '@chakra-ui/react'
import { ReactNode } from 'react'

interface LayoutProps {
  children?: ReactNode
}

export function CommonLayout({ children }: LayoutProps) {
  return (
    <Box
      id="container"
      width="100vw"
      pt={4}
    // sx={{
    //   width: '100%',
    //   flexGrow: 1,
    //   display: 'flex',
    //   overflow: 'hidden',
    //   justifyContent: 'center',
    //   alignItems: 'flex-start',
    //   boxSizing: 'border-box',
    //   padding: '30px 1vw 0 1vw',
    // }}
    >
      {children}
    </Box>
  )
}
