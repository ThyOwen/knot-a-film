//
//  SheetViewGestureState.swift
//  Knot a Film
//
//  Created by Owen O'Malley on 2/15/25.
//


//
//  test.swift
//  Brain
//
//  Created by Owen O'Malley on 2/3/24.
//

import SwiftUI

public enum SheetViewGestureState {
    case inactive
    case active(translationHeight : CGFloat)
    
    var isActive: Bool {
        switch self {
        case .inactive:
            return false
        case .active:
            return true
        }
    }
    
    var translationHeight: CGFloat {
        switch self {
        case .inactive:
            return .zero
        case .active(let height):
            return height
        }
    }
}

struct SheetView<Content: View>: View {
    @Binding var isOpen: Bool
    let maxHeight: CGFloat
    let minHeight : CGFloat
    let content: Content
    
    private let heightDelta : CGFloat
    private let snapDistance : CGFloat
    
    @GestureState private var gestureState: SheetViewGestureState = .inactive
    
    private var gestureIsOpen : Bool {
        if self.isOpen {
            return abs(self.gestureState.translationHeight) < self.snapDistance
        } else {
            return abs(self.gestureState.translationHeight) > self.snapDistance
        }
    }
    
    
    private let bottomRadius: CGFloat = 60
    private let snapRatio: CGFloat = 0.25

    private var baseOffset: CGFloat {
        self.isOpen ? self.heightDelta : 0
    }
    
    private var translationBounds : ClosedRange<CGFloat> {
        self.isOpen ? (-self.heightDelta)...0 : 0...self.heightDelta
    }

    init(isOpen: Binding<Bool>,
         maxHeightFraction: CGFloat,
         minHeight : CGFloat,
         @ViewBuilder content: () -> Content) {
        
        #if os(macOS)
            let maxHeight = (NSScreen.main?.frame.height ?? 400) * maxHeightFraction
        #elseif os(iOS)
            let maxHeight = UIScreen.main.bounds.height * maxHeightFraction
        #endif
        self.maxHeight = maxHeight
        self.minHeight = minHeight
        self.content = content()
        self._isOpen = isOpen
        
        self.heightDelta = maxHeight - minHeight
        
        self.snapDistance = maxHeight * self.snapRatio
    }
    
    var indicator: some View {
        RoundedRectangle(cornerRadius: 16)
            .fill(ThemeColors.secondAccent.opacity(0.2))
            .frame(width: 60, height: 6)
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.4)) {
                    self.isOpen.toggle()
                }
            }
    }
    
    var dragGesture : some Gesture {
        DragGesture()
            .updating(self.$gestureState) { value, state, _ in
                
                let clippedHeight = Self.clamp(value.translation.height, in: self.translationBounds)
                
                //print(value.translation.height)
                
                state = .active(translationHeight: clippedHeight)
            }
            .onEnded { value in
                
                let adjValue = value.translation.height
                
                guard abs(adjValue) > self.snapDistance else {
                    return
                }
                self.isOpen = adjValue > 0
            }
    }
    
    var body: some View {
            ZStack(alignment: .bottom) {
                self.content
                    .frame(height: self.minHeight + Self.clamp(self.baseOffset + self.gestureState.translationHeight, in: 0...self.maxHeight), alignment: .center )
                self.indicator
                    .offset(y: -15)
            }
            .frame(maxWidth: .infinity)
            .background {
                ZStack {
                    
                    UnevenRoundedRectangle(cornerRadii: .init(topLeading: 0,
                                                              bottomLeading: self.bottomRadius,
                                                              bottomTrailing: self.bottomRadius,
                                                              topTrailing: 0),
                                           style: .continuous)
                    .fill(ThemeColors.mainAccent)
                    .padding(.top, self.baseOffset + self.gestureState.translationHeight + (2 * self.bottomRadius))


                    
                    UnevenRoundedRectangle(cornerRadii: .init(topLeading: 0,
                                                              bottomLeading: self.bottomRadius,
                                                              bottomTrailing: self.bottomRadius,
                                                              topTrailing: 0),
                                           style: .continuous)
                    .fill(ThemeColors.mainAccent)

                    }
                }
                .animation(.interactiveSpring, value: self.gestureState.translationHeight)
                .gesture(self.dragGesture)
                .onChange(of: self.gestureIsOpen) { oldValue, newValue in
            }
    }
    
    private static func clamp<T: Comparable>(_ value: T, in range: ClosedRange<T>) -> T {
        return min(max(value, range.lowerBound), range.upperBound)
    }

}

fileprivate struct TestSheetView : View {
    
    @State private var isOpen : Bool = false
    
    var body: some View {
        SheetView(isOpen: self.$isOpen, maxHeightFraction: 0.5, minHeight: 200) {
            ZStack {
                //RoundedRectangle(cornerRadius: 0).fill(Color.red)
                Text("fuck")
            }.frame(maxHeight: .infinity)
        }//.edgesIgnoringSafeArea(.all)
    }
}

#Preview {
    TestSheetView()
}
