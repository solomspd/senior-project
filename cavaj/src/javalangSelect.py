# for i in child.atomic_vals:
#                     if(i):
#                         ast_code.append(i)                        
#                         ast_code.append(' ')
import javalang
def ret_string(mylist):

    x = type(mylist)

    if x == javalang.CompilationUnit:
      return str(x)

    if x == javalang.Import:
        return "import"
      
    if x == javalang.Documented:
        return ""

    if x == javalang.Declaration:
         return str(x[0])
     
    if x == javalang.TypeDeclaration:
        return str(x[0])

    if x == javalang.PackageDeclaration:
        return str(x[0])

    if x == javalang.ClassDeclaration:
        return str(x[0])

    if x == javalang.EnumDeclaration: 
        return str(x[0])

    if x == javalang.InterfaceDeclaration:
        return str(x[0])

    if x == javalang.AnnotationDeclaration:
        return str(x[0])

    if x == javalang.Type:
        return ""

    if x == javalang.BasicType:
        return str(x)

    if x == javalang.ReferenceType:
        return "ref"

    if x == javalang.TypeArgument:
        return ""

    if x == javalang.TypeParameter:
        return str(x)

    if x == javalang.Annotation:
        return str(x)

    if x == javalang.ElementValuePair:
        return "evp"
    
    if x == javalang.ElementArrayValue:
         return "eav"

    if x == javalang.Member:
        return ""

    if x == javalang.MethodDeclaration:
        return str(x[0])

    if x == javalang.FieldDeclaration:
        return str(x[0])

    if x == javalang.ConstructorDeclaration:
        return str(x[0])

    if x == javalang.ConstantDeclaration:
        return str(x[0])

    if x == javalang.ArrayInitializer:
        return "ArIn"

    if x == javalang.VariableDeclaration:
        return str(x[0])

    if x == javalang.LocalVariableDeclaration:
        return str(x[0])

    if x == javalang.VariableDeclarator:
        return x

    if x == javalang.FormalParameter:
        return "param"

    if x == javalang.InferredFormalParameter:
        return "Inferparam"

    if x == javalang.Statement:
        return "Statement"

    if x == javalang.IfStatement:
        return "if"

    if x == javalang.WhileStatement:
         return "while"
     
    if x == javalang.DoStatement:
         return "do"

    if x == javalang.ForStatement:
        return "for"

    if x == javalang.AssertStatement:
        return "assert"

    if x == javalang.BreakStatement:
        return "break"

    if x == javalang.ContinueStatement:
        return "continue"

    if x == javalang.ReturnStatement:
        return "return"

    if x == javalang.ThrowStatement:
        return "throw"

    if x == javalang.SynchronizedStatement:
        return "sync"
        
    if x == javalang.TryStatement:
        return "try"

    if x == javalang.SwitchStatement:
        return "switch"

    if x == javalang.BlockStatement:
        return "block"

    if x == javalang.StatementExpression:
        return "statementExp"

    if x == javalang.TryResource:
        return "try"

    if x == javalang.CatchClause:
        return "catch"

    if x == javalang.CatchClauseParameter:
        return str(x[0])

    if x == javalang.SwitchStatementCase:
        return "caseVars"

    if x == javalang.ForControl:
        return str(x[0])

    if x == javalang.EnhancedForControl:
         return str(x[0])

    # if x == javalang.Expression:

    # if x == javalang.Assignment:

    # if x == javalang.TernaryExpression:

    # if x == javalang.BinaryOperation:

    # if x == javalang.Cast:

    # if x == javalang.MethodReference:

    # if x == javalang.LambdaExpression:

    # if x == javalang.Primary:

    # if x == javalang.Literal:

    # if x == javalang.This:

    # if x == javalang.MemberReference:

    # if x == javalang.Invocation:

    # if x == javalang.ExplicitConstructorInvocation:

    # if x == javalang.SuperConstructorInvocation:

    # if x == javalang.MethodInvocation:

    # if x == javalang.SuperMethodInvocation:

    # if x == javalang.SuperMemberReference:

    # if x == javalang.ArraySelector:

    # if x == javalang.ClassReference:

    # if x == javalang.VoidClassReference:

    # if x == javalang.Creator:

    # if x == javalang.ArrayCreator:

    # if x == javalang.ClassCreator:

    # if x == javalang.InnerClassCreator:

    # if x == javalang.EnumBody:

    # if x == javalang.EnumConstantDeclaration:

    # if x == javalang.AnnotationMethod:
